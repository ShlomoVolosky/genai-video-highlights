from app.config import Config

def test_config_has_minimums():
    # This does not require real env; just checks types and db_url string
    assert isinstance(Config.postgres_port, int)
    url = Config.db_url()
    assert "postgresql+psycopg2://" in url
