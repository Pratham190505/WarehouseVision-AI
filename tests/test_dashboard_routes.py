from fastapi.testclient import TestClient

import app.main as main


def test_all_primary_views_are_directly_routable_and_share_the_shell(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_current_user_optional",
        lambda _request: {"id": "user-1", "name": "Warehouse Operator", "email": "operator@example.com"},
    )
    client = TestClient(main.app)

    for path in ("/dashboard", "/analysis", "/safety", "/box-counting"):
        response = client.get(path)
        assert response.status_code == 200
        assert "WarehouseVision AI" in response.text
        assert 'href="/dashboard" data-route="/dashboard">Dashboard</a>' in response.text
        assert 'href="/analysis" data-route="/analysis">Analysis</a>' in response.text
        assert 'href="/safety" data-route="/safety">Safety' in response.text
        assert 'href="/box-counting" data-route="/box-counting">Box Counting</a>' in response.text
        assert "Warehouse Operator" in response.text
        for view in ("dashboard", "analysis", "safety", "box-counting"):
            assert f'data-view="{view}"' in response.text
        assert 'class="sidebar"' not in response.text


def test_all_primary_views_remain_protected(monkeypatch):
    monkeypatch.setattr(main, "get_current_user_optional", lambda _request: None)
    client = TestClient(main.app)

    for path in ("/dashboard", "/analysis", "/safety", "/box-counting"):
        response = client.get(path, follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/login"
