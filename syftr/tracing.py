import re
import typing as T

import pandas as pd
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)


class InMemorySpanExporter(SpanExporter):
    def __init__(self):
        self.finished_spans = []

    def export(self, spans: T.Sequence[ReadableSpan]) -> SpanExportResult:
        self.finished_spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


def get_span_exporter() -> InMemorySpanExporter:
    resource = Resource(attributes={})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    span_exporter = InMemorySpanExporter()
    span_processor = SimpleSpanProcessor(span_exporter)
    tracer_provider.add_span_processor(span_processor)
    trace_api.set_tracer_provider(tracer_provider=tracer_provider)
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    return span_exporter


def span_to_dataframe(span: ReadableSpan):
    data = {
        "span_name": [span.name],
        "span_start_time": [span.start_time],
        "span_end_time": [span.end_time],
        "span_trace_id": [span.context.trace_id],
    }

    if span.attributes:
        for key, value in span.attributes.items():
            column = f"attribute_{key}"
            if re.search(r".*\.\d+\..*", key):
                tmp = re.sub(r"\.\d+\.", ".", key)
                column = f"attribute_{tmp}"
            data[column] = [value]

    return pd.DataFrame(data)


def spans_to_dataframe(spans: T.Sequence[ReadableSpan]):
    dataframes = [span_to_dataframe(span) for span in spans]
    return pd.concat(dataframes, ignore_index=True)


def _set_tracing_data(df: pd.DataFrame, results: T.Dict[str, T.Any]):
    prefix = "trace__"
    num_decimals = 2

    results[prefix + "num_trace_ids"] = len(df["span_trace_id"].unique())

    df["span_duration"] = df["span_end_time"] - df["span_start_time"]
    trace_avg_durations = df.groupby("span_name")["span_duration"].mean().to_dict()
    trace_avg_durations = {
        f"{prefix}{key.replace('__', '')}__duration_mean": round(value, num_decimals)
        for key, value in trace_avg_durations.items()
    }
    results.update(trace_avg_durations)

    trace_total_durations = df.groupby("span_name")["span_duration"].sum().to_dict()
    trace_total_durations = {
        f"{prefix}{key.replace('__', '')}__duration_total": value
        for key, value in trace_total_durations.items()
    }
    results.update(trace_total_durations)

    df["input_length"] = df["attribute_input.value"].apply(
        lambda x: len(x) if isinstance(x, str) else 0
    )
    df["output_lenght"] = df["attribute_output.value"].apply(
        lambda x: len(x) if isinstance(x, str) else 0
    )

    trace_avg_input_lengths = df.groupby("span_name")["input_length"].mean().to_dict()
    trace_avg_input_lengths = {
        f"{prefix}{key.replace('__', '')}__input_length_mean": round(
            value, num_decimals
        )
        for key, value in trace_avg_input_lengths.items()
    }
    results.update(trace_avg_input_lengths)

    trace_avg_ouptput_lengths = (
        df.groupby("span_name")["output_lenght"].mean().to_dict()
    )
    trace_avg_ouptput_lengths = {
        f"{prefix}{key.replace('__', '')}__output_length_mean": round(
            value, num_decimals
        )
        for key, value in trace_avg_ouptput_lengths.items()
    }
    results.update(trace_avg_ouptput_lengths)


def set_tracing_metrics(
    span_exporter: InMemorySpanExporter, results: T.Dict[str, T.Any]
):
    spans = span_exporter.finished_spans
    if not spans:
        return

    df = spans_to_dataframe(spans)
    _set_tracing_data(df, results)
