from unstructured.partition.pdf import partition_pdf

# Check valid chunking strategies
print(partition_pdf.__annotations__["chunking_strategy"])
# Expected Output: typing.Literal['auto', 'fast', 'hi_res', 'by_title']