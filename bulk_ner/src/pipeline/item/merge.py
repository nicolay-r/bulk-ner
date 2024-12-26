from bulk_ner.src.pipeline.item.base import BasePipelineItem


class MergeTextEntries(BasePipelineItem):

    def apply(self, input_data, pipeline_ctx=None):
        assert (isinstance(input_data, list))
        buffer = []

        out = []
        for entry in input_data:
            if isinstance(entry, str):
                buffer.append(entry)
            else:
                # release buffer.
                if len(buffer) > 0:
                    out.append(" ".join(buffer))
                    buffer.clear()
                out.append(entry)

        # release buffer.
        if len(buffer) > 0:
            out.append(" ".join(buffer))
            buffer.clear()

        return out
