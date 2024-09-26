from arekit.common.pipeline.items.base import BasePipelineItem


class HandleListPipelineItem(BasePipelineItem):

    def __init__(self, map_item_func, filter_item_func=None, **kwargs):
        assert(callable(map_item_func))
        super(HandleListPipelineItem, self).__init__(**kwargs)
        self.__map_item = map_item_func
        self.__filter_func = filter_item_func

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, list))

        l = []
        for i, term in enumerate(input_data):
            if self.__filter_func is not None and not self.__filter_func(term):
                continue
            l.append(self.__map_item(i, term))

        return l
