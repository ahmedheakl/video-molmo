from bitsandbytes.optim import PagedAdamW32bit
from transformers import get_cosine_schedule_with_warmup

def build_optimizer(model, num_warmup_steps, num_training_steps):
    lr_llm       = 1e-5   # LLM core
    lr_encoder   = 5e-6   # visual encoder
    lr_connector = 5e-6   # multimodal adapter

    param_groups = [
        {
            "params": [p for n,p in model.named_parameters() if n.startswith("model.transformer")],
            "lr": lr_llm,
            "name": "llm"
        },
        {
            "params": [p for n,p in model.named_parameters() if n.startswith("model.vision_backbone.image_vit")],
            "lr": lr_encoder,
            "name": "encoder"
        },
        {
            "params": [p for n, p in model.named_parameters() if any(s in n for s in ["image_pooling_2d", "adapter", "projector"])],
            "lr": lr_connector,
            "name": "connector"
        },
    ]

    # 2) Build the optimizer
    optimizer = PagedAdamW32bit(
        param_groups,
        betas=(0.9, 0.95),
        eps=1e-6,
        weight_decay=0.1
    )

    # 3) Build the cosine-with-warmup scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    return optimizer, scheduler