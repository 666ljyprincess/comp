import smart_community_perception.announce_node as an


def test_speak_falls_back_to_log_when_no_tts(monkeypatch):
    monkeypatch.setattr(an, "_TTS_AVAILABLE", False)
    assert an.speak("前方红灯") == "log"


def test_speak_uses_tts_when_available(monkeypatch):
    calls = []
    monkeypatch.setattr(an, "_TTS_AVAILABLE", True)
    monkeypatch.setattr(an, "_say_tts", lambda t: calls.append(t))
    assert an.speak("绿灯通行") == "tts"
    assert calls == ["绿灯通行"]
