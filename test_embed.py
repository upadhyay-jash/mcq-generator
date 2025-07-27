from langchain_openai import AzureOpenAIEmbeddings
import config

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=config.AZURE_DEPLOYMENT_NAME_EMBEDDING,
    azure_endpoint=config.AZURE_OPENAI_ENDPOINT,            # ✅ use this now
    openai_api_key=config.AZURE_OPENAI_API_KEY,
    openai_api_version=config.AZURE_API_VERSION,
    chunk_size=1000
)

res = embeddings.embed_documents(["Test sentence 1", "Another one"])
print(res)
