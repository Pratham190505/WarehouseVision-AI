from fastapi.testclient import TestClient
import re

import app.main as main


def test_all_primary_views_are_directly_routable_and_share_the_shell(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_current_user_optional",
        lambda _request: {"id": "user-1", "name": "Warehouse Operator", "email": "operator@example.com"},
    )
    client = TestClient(main.app)

    for path in ("/dashboard", "/analysis", "/safety", "/box-counting", "/settings"):
        response = client.get(path)
        assert response.status_code == 200
        assert "WarehouseVision AI" in response.text
        assert 'href="/dashboard" data-route="/dashboard"' in response.text
        assert 'href="/analysis" data-route="/analysis"' in response.text
        assert 'href="/safety" data-route="/safety"' in response.text
        assert 'href="/box-counting" data-route="/box-counting"' in response.text
        assert "Warehouse Operator" in response.text
        nav_start = response.text.index('<nav class="primary-navigation"')
        nav_end = response.text.index('</nav>', nav_start)
        assert 'href="/settings" data-route="/settings"' in response.text[nav_start:nav_end]
        nav_markup = response.text[nav_start:nav_end]
        assert 'aria-label="Settings"' in nav_markup
        active_anchor = next(
            anchor for anchor in re.findall(r'<a class="nav-link[^>]*>.*?</a>', nav_markup, re.S)
            if f'href="{path}"' in anchor
        )
        assert 'active' in active_anchor.split('href=')[0]
        assert 'aria-current="page"' in active_anchor
        for view in ("dashboard", "analysis", "safety", "box-counting", "settings"):
            assert f'data-view="{view}"' in response.text
        assert 'href="/settings" data-route="/settings"' in response.text
        assert 'id="settingsForm"' in response.text
        assert 'id="settingsModalOverlay"' not in response.text
        assert 'class="sidebar"' not in response.text


def test_all_primary_views_remain_protected(monkeypatch):
    monkeypatch.setattr(main, "get_current_user_optional", lambda _request: None)
    client = TestClient(main.app)

    for path in ("/dashboard", "/analysis", "/safety", "/box-counting", "/settings"):
        response = client.get(path, follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/login"
