""".github/workflows/report.yml: its inputs, how they reach the shell, the secrets, the caches and publishing.

Text-level checks (no YAML parser needed): the workflow is what runs the user's reports from a phone, so a broken
input or a secret on a command line would only show up on GitHub.
"""

import json
import os
import re
import subprocess
import sys
import textwrap
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
    assert WORKFLOW.count("secrets.LICHESS_TOKEN") == 1
    # the key itself reaches only the analysis; the install step learns only whether it is set
    assert WORKFLOW.count("secrets.ANTHROPIC_API_KEY }}") == 1
    assert WORKFLOW.count("secrets.ANTHROPIC_API_KEY") == 2
    assert "LLM_KEY_SET: ${{ secrets.ANTHROPIC_API_KEY != '' }}" in block("Install chess-insights")


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
    for name in ("Restore game and engine cache", "Save game and engine cache"):
        step = block(name)
        assert "path: .chess-insights-cache\n" in step and "!" not in step.split("with:", 1)[1]
    assert "--cache-dir .chess-insights-cache " in block("Analyse games")


def test_the_game_and_engine_cache_is_saved_even_when_the_analysis_fails_or_times_out():
    """actions/cache@v4 saves only when every step succeeded: a timeout (or a rejected push) lost all of Stockfish's
    work, and every re-run started from zero. Restore and save are now separate, the save runs always, right after
    the analysis, and the analysis has its own time limit so the save still fits in the job."""
    restore, analyse, save = (block(n) for n in (
        "Restore game and engine cache", "Analyse games", "Save game and engine cache"))
    assert "uses: actions/cache/restore@v4" in restore and "uses: actions/cache/save@v4" in save
    assert "uses: actions/cache@v4" not in WORKFLOW
    assert "if: always()" in save
    order = [WORKFLOW.index(f"- name: {n}") for n in (
        "Restore game and engine cache", "Analyse games", "Save game and engine cache", "Attach the report to this run",
        "Save the report on the reports branch")]
    assert order == sorted(order)
    key = re.search(r"\n\s+key: (.*)\n", restore).group(1)
    assert f"key: {key}\n" in save  # the same key: one run, one saved cache
    job = int(re.search(r"\n    timeout-minutes: (\d+)\n", WORKFLOW).group(1))
    step = int(re.search(r"timeout-minutes: (\d+)", analyse).group(1))
    download = int(re.search(r"timeout-minutes: (\d+)", block("Download the puzzle database (once a month)")).group(1))
    assert job == 120 and step + download + 10 <= job  # setup, the monthly download and the saving still fit
    assert "id: analyse" in analyse
    carry_on = block("Say how to carry on")
    assert "if: failure() && steps.analyse.outcome == 'failure'" in carry_on and "carries on" in carry_on


def test_the_cache_key_is_lowercase_and_cannot_match_a_longer_username():
    """A restore key is a prefix: 'chess-insights-bob-' also matched bob-smith's cache. The key now ends the name with
    '.', which a chess.com username cannot hold, and keeps the old key as a second restore key so the caches saved
    under it (chess-insights-BigMuffEater-<run>) still restore once."""
    from chess_insights import api

    restore, save = block("Restore game and engine cache"), block("Save game and engine cache")
    key = "chess-insights-${{ matrix.cache_name }}.${{ github.run_id }}-${{ github.run_attempt }}"
    assert f"key: {key}\n" in restore and f"key: {key}\n" in save
    keys = [k.strip() for k in re.search(r"restore-keys: \|\n((?:\s{12}.*\n)+)", restore).group(1).splitlines()]
    assert keys == ["chess-insights-${{ matrix.cache_name }}.", "chess-insights-${{ matrix.username }}-"]
    assert api._USERNAME_RE.match("bob.smith") is None and api._USERNAME_RE.match("bob-smith")


def players_script() -> str:
    """The players job's Python script (the heredoc), as the runner gets it."""
    m = re.search(r"python3 - <<'EOF'\n(.*?)\n\s*EOF\n", WORKFLOW, re.S)
    return textwrap.dedent(m.group(1))


def run_players(tmp_path, players="", **env):
    (tmp_path / "players.txt").write_text(players, encoding="utf-8")
    out = tmp_path / "github_output"
    out.write_text("", encoding="utf-8")
    base = {"EVENT": "push", "IN_USER": "", "IN_TZ": "", "IN_ENGINE": "true", "IN_ENGINE_GAMES": "300",
            "IN_DEPTH": "12", "GITHUB_OUTPUT": str(out), "PATH": os.environ.get("PATH", "")}
    proc = subprocess.run([sys.executable, "-c", players_script()], cwd=tmp_path, env={**base, **env},
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    lines = dict(line.split("=", 1) for line in out.read_text(encoding="utf-8").splitlines())
    return json.loads(lines["matrix"])["include"], proc.stdout


def test_the_players_step_names_each_cache_in_lowercase(tmp_path):
    rows, _ = run_players(tmp_path, "BigMuffEater America/New_York\n# a comment\nBob-Smith\n")
    assert rows == [
        {"username": "BigMuffEater", "timezone": "America/New_York", "cache_name": "bigmuffeater"},
        {"username": "Bob-Smith", "timezone": "", "cache_name": "bob-smith"},
    ]
    rows, _ = run_players(tmp_path, EVENT="workflow_dispatch", IN_USER=" Weird.Name!  ", IN_TZ="")
    assert rows == [{"username": "Weird.Name!", "timezone": "", "cache_name": "weirdname"}]


@pytest.mark.parametrize("games, depth, engine, warned", [
    ("300", "12", "true", False),  # the defaults: about 3 minutes of Stockfish
    ("all", "12", "true", False),  # about 35 minutes for 3,400 games
    ("all", "15", "true", True),  # BigMuffEater's 3,354 games at depth 15: about 107 minutes
    ("1000", "15", "true", False),  # the run that took 32 minutes
    ("3000", "15", "true", True),
    ("all", "18", "false", False),  # no engine, no warning
])
def test_the_players_step_warns_about_runs_that_may_not_fit(tmp_path, games, depth, engine, warned):
    _, stdout = run_players(tmp_path, "BigMuffEater\n", IN_ENGINE_GAMES=games, IN_DEPTH=depth, IN_ENGINE=engine)
    assert ("::warning title=A long engine run::" in stdout) is warned
    if warned:
        assert f"depth {depth}" in stdout and "cached" in stdout and "80 minutes" in stdout


def test_the_ai_coach_sdk_is_installed_only_when_its_key_is_set():
    """--coach-llm needs the Anthropic SDK (the llm extra); the install step gets only whether the secret is set."""
    install = block("Install chess-insights")
    script = install.split("run: |", 1)[1]
    assert 'if [ "$LLM_KEY_SET" = "true" ]; then pip install -e ".[llm]"; else pip install -e .; fi' in script
    assert "ANTHROPIC_API_KEY:" not in install  # never the key itself in pip's environment
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^llm = \["anthropic', pyproject, re.M)
    assert WORKFLOW.index("- name: Install chess-insights") < WORKFLOW.index("- name: Analyse games")


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


def test_publishing_retries_when_another_run_pushed_first():
    """Two runs (a players.txt push and a manual run) can publish at once: the second push is rejected. It now
    rebases onto the other run's commit and tries again, instead of failing the job."""
    publish = block("Save the report on the reports branch")
    loop = re.search(r"for attempt in 1 2 3; do\n(.*?)\n\s*done\n\s*exit 1", publish, re.S)
    assert loop, publish
    assert 'if git -C "$dest" push --quiet origin reports; then exit 0; fi' in loop.group(1)
    assert 'git -C "$dest" pull --quiet --rebase origin reports' in loop.group(1)


def test_players_txt_gives_the_time_zone():
    lines = [line.split("#", 1)[0].split() for line in (ROOT / "players.txt").read_text(encoding="utf-8").splitlines()]
    assert ["BigMuffEater", "America/New_York"] in lines
