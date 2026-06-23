from torch.utils.data import Dataset

from molt.utils.vlm_utils import should_expand_image_placeholder, split_image_placeholder


def preprocess_data(
    data,
    input_template=None,
    input_key="input",
    label_key=None,
    apply_chat_template=None,
    expand_image_placeholder: bool = False,
    tools=None,
) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        if expand_image_placeholder:
            # Qwen2.5-VL / Qwen3.6-VL chat templates iterate over structured
            # content lists `[{"type": "image"}, {"type": "text", "text": ...}]`
            # and emit a model-specific placeholder (e.g. `<|image_pad|>`) that
            # vLLM's multimodal processor recognizes for prompt replacement.
            # Datasets store user content as a single string `"<image>\nproblem"`,
            # so split on each `<image>` and convert to the structured form.
            chat = [split_image_placeholder(msg) for msg in chat]
        # Otherwise: pass content through verbatim. Chat templates that emit
        # `message.content | string` won't iterate structured content; the
        # literal `<image>` in the text is what the processor matches on.
        # `tools` (OpenAI function-call schemas) are rendered by Qwen / Llama
        # chat templates into a system-side preamble that teaches the model
        # the `<tool_call>{...}</tool_call>` emission format natively.
        kwargs = {}
        if tools:
            kwargs["tools"] = tools
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True, **kwargs)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)

    # Verifier ground-truth answer for RL reward computation (empty if no label_key).
    label = "" if label_key is None else data[label_key]
    return prompt, label


class PromptDataset(Dataset):
    """
    Dataset for policy RL prompts.

    Args:
        dataset: prompt dataset
        tokenizer: tokenizer for the actor model
        strategy: training strategy; supplies the ``args.data`` column keys
        input_template: optional ``str.format`` template applied to each prompt
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.input_template = input_template

        # LAZY rendering: keep the (memory-mapped Arrow) dataset and apply_chat_template each row
        # ON DEMAND in __getitem__, instead of eagerly rendering EVERY row into Python lists here.
        # Eager preprocessing held the entire rendered corpus in RAM (self.prompts/labels/images),
        # which OOMs and stalls __init__ on large datasets (e.g. multi-million-row, ~100KB/prompt
        # multimodal corpora). Lazy keeps init O(1) memory + near-instant; the dataloader workers
        # render only the rows actually sampled. __getitem__'s return value is unchanged.
        self.dataset = dataset
        self.input_key = getattr(self.strategy.args.data, "input_key", None)
        self.label_key = getattr(self.strategy.args.data, "label_key", None)
        self.tools_key = getattr(self.strategy.args.data, "tools_key", None)
        self.image_key = getattr(self.strategy.args.data, "image_key", "images")
        apply_chat_template = getattr(self.strategy.args.data, "apply_chat_template", False)
        self.apply_chat_template = self.tokenizer.apply_chat_template if apply_chat_template else None
        self.expand_image_placeholder = should_expand_image_placeholder(self.tokenizer)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Render this row's prompt on demand (lazy). The Arrow-backed dataset is memory-mapped,
        # so indexing reads a single row without materializing the corpus.
        data = self.dataset[idx]
        prompt, label = preprocess_data(
            data,
            self.input_template,
            self.input_key,
            self.label_key,
            self.apply_chat_template,
            expand_image_placeholder=self.expand_image_placeholder,
            tools=data.get(self.tools_key) if self.tools_key else None,
        )
        return data.get("datasource", "default"), prompt, label, data.get(self.image_key, None)

    def collate_fn(self, item_list):
        datasources = []
        prompts = []
        labels = []
        images = []
        for datasource, prompt, label, img in item_list:
            datasources.append(datasource)
            prompts.append(prompt)
            labels.append(label)
            images.append(img)

        return datasources, prompts, labels, images
