""".github/workflows/report.yml: its inputs, how they reach the shell, the secrets, the puzzle cache and publishing.

Text-level checks (no YAML parser needed): the workflow is what runs the user's reports from a phone, so a broken
input or a secret on a command line would only show up on GitHub.
"""

import re
from pathlib import Path

import pytest

from chess_insights import cli

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = (ROOT / ".github" / "workflows" / "report.yml").read_text(encoding="utf-8")


def block(name: str) -> str:
    """The text of one step, from its '- name:' line to the next step."""
    start = WORKFLOW.index(f"- name: {name}")
    nxt = WORKFLOW.find("\n      - ", start + 1)
    return WORKFLOW[start: nxt if nxt != -1 else len(WORKFLOW)]


def run_scripts() -> list[str]:
    """Every multi-line ``run: |`` script."""
    return re.findall(r"run: \|\n((?:\s{10,}.*\n?)+)", WORKFLOW)


def test_triggers_and_jobs_are_unchanged():
    assert "workflow_dispatch:" in WORKFLOW and "paths: [players.txt]" in WORKFLOW
    assert "jobs:\n  players:" in WORKFLOW and "\n  report:\n    needs: players" in WORKFLOW
    assert "max-parallel: 1" in WORKFLOW and "contents: write" in WORKFLOW


@pytest.mark.parametrize("name, default", [
    ("username", None), ("timezone", '""'), ("engine", "true"), ("engine_games", '"300"'), ("depth", '"12"'),
    ("engine_sample", "balanced"), ("coach", "true"), ("coach_depth", '"20"'), ("coach_max", '"150"'),
])
def test_dispatch_inputs(name, default):
    m = re.search(rf"\n      {name}:\n((?:        .*\n)+)", WORKFLOW)
    assert m, name
    if default is not None:
        assert f"default: {default}" in m.group(1)


def test_choice_and_boolean_inputs():
    assert re.search(r"engine_sample:\n(?:        .*\n)*        type: choice\n        options: \[balanced, recent\]",
                     WORKFLOW)
    assert re.search(r"coach:\n(?:        .*\n)*        type: boolean", WORKFLOW)


def test_players_txt_runs_use_the_same_defaults():
    """A push has no inputs: the env lines fall back to the dispatch defaults (balanced sample, coaching on)."""
    assert "ENGINE_SAMPLE: ${{ inputs.engine_sample || 'balanced' }}" in WORKFLOW
    assert "COACH: ${{ github.event_name != 'workflow_dispatch' || inputs.coach }}" in WORKFLOW
    assert "COACH_DEPTH: ${{ inputs.coach_depth || '20' }}" in WORKFLOW
    assert "COACH_MAX: ${{ inputs.coach_max || '150' }}" in WORKFLOW


def test_inputs_and_secrets_reach_the_shell_only_through_the_environment():
    scripts = run_scripts()
    assert len(scripts) >= 4 and any("chess-insights" in s for s in scripts)
    for script in scripts:
        assert "${{" not in script, script
    assert "${{" not in block("Download the puzzle database (once a month)").split("run:", 1)[1]
    analyse = block("Analyse games")
    assert "LICHESS_TOKEN: ${{ secrets.LICHESS_TOKEN }}" in analyse
    assert "ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}" in analyse
    assert "--lichess-token" not in WORKFLOW  # the CLI reads it from the environment
    assert WORKFLOW.count("secrets.ANTHROPIC_API_KEY") == WORKFLOW.count("secrets.LICHESS_TOKEN") == 1


def test_analyse_step_passes_the_new_options():
    analyse = block("Analyse games")
    assert '--engine-sample "$ENGINE_SAMPLE"' in analyse
    coach = re.search(r'if \[ "\$COACH" = "true" \]; then\n(.*?)\n\s*fi\n\s*fi', analyse, re.S)
    assert coach, "--coach only inside the engine branch"
    assert '--coach --coach-depth "$COACH_DEPTH" --coach-max "$COACH_MAX"' in coach.group(1)
    assert 'if [ -n "$ANTHROPIC_API_KEY" ]; then args+=(--coach-llm); fi' in coach.group(1)
    assert analyse.index('if [ "$ENGINE" = "true" ]') < analyse.index("--coach ")
    assert 'args+=(--previous "$PREVIOUS_REPORT")' in analyse


def test_every_option_the_workflow_passes_exists():
    options = set(re.findall(r"(--[a-z][a-z0-9-]+)", block("Analyse games")))
    known = {a for action in cli.build_parser()._subparsers._group_actions[0].choices["report"]._actions
             for a in action.option_strings}
    assert options <= known, options - known


def test_the_game_and_engine_cache_path_is_unchanged():
    """actions/cache keys a cache's version on its paths: changing them would make every cache saved by earlier runs
    unrestorable, so the first run after an upgrade would download every game and re-run Stockfish from scratch."""
    restore = block("Restore game and engine cache")
    assert "path: .chess-insights-cache\n" in restore and "!" not in restore.split("with:", 1)[1]
    assert "--cache-dir .chess-insights-cache " in block("Analyse games")


def test_previous_report_comes_from_the_reports_branch():
    fetch = block("Fetch the previous report")
    assert "git fetch --quiet --depth 1 origin reports" in fetch
    assert '"FETCH_HEAD:$name/latest/$name.json"' in fetch
    assert 'echo "PREVIOUS_REPORT=$previous" >> "$GITHUB_ENV"' in fetch
    assert WORKFLOW.index("Fetch the previous report") < WORKFLOW.index("- name: Analyse games")


def test_puzzle_database_is_downloaded_at_most_once_a_month_and_never_blocks():
    assert 'echo "month=$(date -u +%Y-%m)" >> "$GITHUB_OUTPUT"' in WORKFLOW
    restore, download, save = (block(n) for n in (
        "Restore the puzzle database", "Download the puzzle database (once a month)", "Save the puzzle database"))
    assert "actions/cache/restore@v4" in restore and "key: lichess-puzzles-${{ steps.month.outputs.month }}" in restore
    assert "path: .chess-insights-puzzles\n" in restore and "path: .chess-insights-puzzles\n" in save
    assert "steps.puzzles.outputs.cache-hit != 'true'" in download and "continue-on-error: true" in download
    assert "puzzles-db --cache-dir .chess-insights-puzzles" in download
    assert "actions/cache/save@v4" in save and "steps.puzzles_download.outcome == 'success'" in save
    # the analysis finds it only when it is there (a failed first download leaves the drills on Lichess's pages)
    assert "if [ -d .chess-insights-puzzles ]; then args+=(--puzzle-db .chess-insights-puzzles); fi" in block(
        "Analyse games")
    order = [WORKFLOW.index(f"- name: {n}") for n in ("Restore the puzzle database",
             "Download the puzzle database (once a month)", "Save the puzzle database", "Analyse games")]
    assert order == sorted(order)


def test_publishing_copies_every_file_the_run_wrote():
    publish = block("Save the report on the reports branch")
    assert 'cp -R reports/. "$dest/$REPORT_NAME/$stamp/"' in publish
    assert 'cp -R reports/. "$dest/$REPORT_NAME/latest/"' in publish
    assert "path: reports/" in block("Attach the report to this run")


def test_players_txt_gives_the_time_zone():
    lines = [line.split("#", 1)[0].split() for line in (ROOT / "players.txt").read_text(encoding="utf-8").splitlines()]
    assert ["BigMuffEater", "America/New_York"] in lines
