from memk.core.metrics import MetricsCollector, RequestMetrics


def test_metrics_summary_tracks_error_rate():
    collector = MetricsCollector()

    collector.record_request(RequestMetrics(
        operation="GET /ok",
        latency_ms=10,
        cache_hit=False,
        degraded=False,
        status_code=200,
    ))
    collector.record_request(RequestMetrics(
        operation="GET /fail",
        latency_ms=20,
        cache_hit=False,
        degraded=True,
        status_code=500,
        error=True,
    ))

    summary = collector.get_metrics_summary()

    assert summary["requests"]["total"] == 2
    assert summary["errors"]["total"] == 1
    assert summary["errors"]["rate"] == 0.5
    assert summary["degraded"]["total"] == 1
