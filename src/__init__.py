import warnings

# milvus-lite (pulled in by pymilvus for local-mode Milvus) imports the
# deprecated pkg_resources API at construction time. There's no fix on our
# side short of jumping pymilvus to an untested major version, so silence
# this one specific warning rather than the whole UserWarning category.
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
    module="milvus_lite",
)
