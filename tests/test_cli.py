from model_structure_viewer import cli


def test_cli_preserves_offline_environment_without_flag(monkeypatch):
    seen = []
    monkeypatch.setenv("MSV_OFFLINE", "1")
    monkeypatch.setattr(cli, "cmd_list", lambda args, settings: seen.append(settings) or 0)

    assert cli.main(["list"]) == 0
    assert seen[0].offline is True


def test_cli_offline_flag_enables_offline_mode(monkeypatch):
    seen = []
    monkeypatch.setenv("MSV_OFFLINE", "0")
    monkeypatch.setattr(cli, "cmd_list", lambda args, settings: seen.append(settings) or 0)

    assert cli.main(["--offline", "list"]) == 0
    assert seen[0].offline is True
