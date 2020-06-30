from src.claim_retriever import NounFrameClaimRetriever
from src.embedder.model import DummyEmbeddingModel
from src.entities.textpieces import Paragraph

retriever = NounFrameClaimRetriever(embedding_model=DummyEmbeddingModel())

p = "However, voters decided that if the stadium was such a good idea someone would build it himself, " \
    "and rejected it 59% to 41%."
retriever.add(Paragraph.from_text(p, 0, 1))

request = "Voting about the stadium will take place on Friday"
retriever.retrieve(request)

print("done")
