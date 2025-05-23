import subprocess
from typing import NamedTuple

from finn.utils import info


def test_citation_text():
    assert isinstance(info.citation_text, str)
    assert "doi" in info.citation_text


def test_linux_os_name_file(monkeypatch, tmp_path):
    with open(tmp_path / "os-release", "w") as f_p:
        f_p.write('PRETTY_NAME="Test text"\n')

    monkeypatch.setattr(info, "OS_RELEASE_PATH", str(tmp_path / "os-release"))

    assert info._linux_sys_name() == "Test text"

    with open(tmp_path / "os-release", "w") as f_p:
        f_p.write('NAME="Test2"\nVERSION="text"')

    assert info._linux_sys_name() == "Test2 text"

    with open(tmp_path / "os-release", "w") as f_p:
        f_p.write('NAME="Test2"\nVERSION_ID="text2"')

    assert info._linux_sys_name() == "Test2 text2"

    with open(tmp_path / "os-release", "w") as f_p:
        f_p.write('NAME="Test2"\nVERSION="text"\nVERSION_ID="text2"')

    assert info._linux_sys_name() == "Test2 text"

    with open(tmp_path / "os-release", "w") as f_p:
        f_p.write(
            'PRETTY_NAME="Test text"\nNAME="Test2"\nVERSION="text"\nVERSION_ID="text2"'
        )

    assert info._linux_sys_name() == "Test text"


class _CompletedProcessMock(NamedTuple):
    stdout: bytes


def _lsb_mock(*_args, **_kwargs):
    return _CompletedProcessMock(
        stdout=b"Description:	Ubuntu Test 20.04\nRelease:	20.04"
    )


def _lsb_mock2(*_args, **_kwargs):
    return _CompletedProcessMock(stdout=b"Description:	Ubuntu Test\nRelease:	20.05")


def test_linux_os_name_lsb(monkeypatch, tmp_path):
    monkeypatch.setattr(info, "OS_RELEASE_PATH", str(tmp_path / "os-release"))
    monkeypatch.setattr(subprocess, "run", _lsb_mock)
    assert info._linux_sys_name() == "Ubuntu Test 20.04"
    monkeypatch.setattr(subprocess, "run", _lsb_mock2)
    assert info._linux_sys_name() == "Ubuntu Test 20.05"
