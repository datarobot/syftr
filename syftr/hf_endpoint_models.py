from syftr.configuration import cfg
from syftr.hf_endpoint_embeddings import HFEndpointEmbeddings


def get_hf_endpoint_embed_model(model_name: str) -> HFEndpointEmbeddings | None:
    embed_model_cfg = cfg.hf_embeddings.models_config_map.get(model_name, None)
    if not embed_model_cfg:
        raise ValueError(f"Model {model_name} not found in config")
    embed_model = HFEndpointEmbeddings(
        hf_api_key=str(cfg.hf_embeddings.api_key.get_secret_value()),
        hf_api_url=embed_model_cfg.api_url,
        model_name=embed_model_cfg.embedding_model_name,
        max_length=embed_model_cfg.max_length,
        hf_embedding_batch_size=embed_model_cfg.hf_embedding_batch_size,
        query_prefix=embed_model_cfg.query_prefix,
        text_prefix=embed_model_cfg.text_prefix,
    )

    return embed_model
