from services.ingest import DoclingChunker, DoclingParser, IngestionPipeline, make_source_document


def main():
    doc = make_source_document("ALICE_ingest_data_flow.pdf")
    pipeline = IngestionPipeline(parser=DoclingParser(), chunker=DoclingChunker())
    chunks = pipeline.run(doc)

    for i, chunk in enumerate(chunks):
        print(f"[{i}] heading={chunk.provenance.section_heading!r} page={chunk.provenance.page_number}")
        print(f"    {chunk.content[:80]!r}")


if __name__ == "__main__":
    main()
