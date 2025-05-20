from llama_index.core.instrumentation import get_dispatcher
from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.llama_index._handler import EventHandler
from openinference.instrumentation.llama_index.version import __version__
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from syftr.configuration import cfg


def instrument_arize(endpoint: str = cfg.instrumentation.otel_endpoint) -> None:
    """Arize global handler conflicts with our token tracker.

    This accomplishes the same thing.

    Also compatible with OTEL tracing, eg. otel-collector
    """
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

    tracer = OITracer(
        trace_api.get_tracer(__name__, __version__, tracer_provider),
        config=TraceConfig(),
    )

    _event_handler = EventHandler(tracer=tracer)
    _span_handler = _event_handler._span_handler
    dispatcher = get_dispatcher()
    for span_handler in dispatcher.span_handlers:
        if span_handler.__class__.__name__ == "_SpanHandler":
            break
    else:
        dispatcher.add_span_handler(_span_handler)
    for event_handler in dispatcher.event_handlers:
        if event_handler.__class__.__name__ == "EventHandler":
            break
    else:
        dispatcher.add_event_handler(_event_handler)
