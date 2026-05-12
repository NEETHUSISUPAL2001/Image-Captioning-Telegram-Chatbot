import os
import logging
from PIL import Image
from io import BytesIO

import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

from transformers import BlipProcessor, BlipForConditionalGeneration

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is not set.\nRun: export TELEGRAM_BOT_TOKEN='your_token_here'  then try again.")
BLIP_MODEL = "Salesforce/blip-image-captioning-large"

# ── Load BLIP model (once at startup) ────────────────────────────────────────
logger.info("Loading BLIP model — this may take a moment…")
processor = BlipProcessor.from_pretrained(BLIP_MODEL)
model     = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)
logger.info("BLIP model loaded successfully.")

# ── Helpers ───────────────────────────────────────────────────────────────────
def generate_caption(image: Image.Image, conditional_text: str = "") -> str:
    """Run BLIP inference and return a high-quality caption string."""
    if conditional_text:
        inputs = processor(image, conditional_text, return_tensors="pt")
    else:
        inputs = processor(image, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_new_tokens=60,
        min_length=10,
        num_beams=5,
        length_penalty=1.2,
        repetition_penalty=1.5,
        early_stopping=True,
    )
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# ── Handlers ──────────────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Welcome message."""
    text = (
        "👋 *Welcome to the AI Image Caption Bot!*\n\n"
        "I use the *BLIP* Vision–Language Model to understand images and generate human-like captions.\n\n"
        "📸 Simply *send me any photo* and I'll describe what I see.\n"
        "You can also add a caption hint with your image (e.g. _'a photo of'_).\n\n"
        "Type /help for more information."
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Help message."""
    text = (
        "🤖 *AI Image Caption Bot — Help*\n\n"
        "*Commands:*\n"
        "  /start — Welcome message\n"
        "  /help  — This help text\n"
        "  /about — About the technology used\n\n"
        "*Usage:*\n"
        "  • Send any image and the bot will caption it automatically.\n"
        "  • Include text with your image to guide captioning (e.g. _'a photo of'_).\n\n"
        "*Tips:*\n"
        "  • Clear, well-lit images produce better captions.\n"
        "  • The model works best with real-world photos."
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """About message."""
    text = (
        "ℹ️ *About this Bot*\n\n"
        "This bot uses *BLIP* (Bootstrapping Language-Image Pre-training), "
        "a Transformer-based Vision Language Model (VLM) developed by Salesforce.\n\n"
        "🔬 *Technology Stack:*\n"
        "  • `BLIP` — Feature extraction & caption generation\n"
        "  • `HuggingFace Transformers` — Model loading & inference\n"
        "  • `python-telegram-bot` — Telegram integration\n"
        "  • `Pillow` — Image preprocessing\n\n"
        "📚 Built with Deep Learning & NLP."
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Receive a photo, run BLIP, return the caption."""
    await update.message.reply_text("🔍 Analysing your image, please wait…")

    try:
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        image = Image.open(BytesIO(photo_bytes)).convert("RGB")

        conditional_text = update.message.caption or ""
        caption = generate_caption(image, conditional_text)

        reply = f"🖼️ *Caption:*\n_{caption}_"
        await update.message.reply_text(reply, parse_mode="Markdown")

    except Exception as exc:
        logger.error("Error processing photo: %s", exc)
        await update.message.reply_text("❌ Sorry, something went wrong while processing your image. Please try again.")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle images sent as documents (uncompressed)."""
    doc = update.message.document
    if not doc.mime_type or not doc.mime_type.startswith("image/"):
        await update.message.reply_text("⚠️ Please send an image file (JPEG, PNG, etc.).")
        return

    await update.message.reply_text("🔍 Analysing your image, please wait…")

    try:
        doc_file = await doc.get_file()
        doc_bytes = await doc_file.download_as_bytearray()
        image = Image.open(BytesIO(doc_bytes)).convert("RGB")

        conditional_text = update.message.caption or ""
        caption = generate_caption(image, conditional_text)

        reply = f"🖼️ *Caption:*\n_{caption}_"
        await update.message.reply_text(reply, parse_mode="Markdown")

    except Exception as exc:
        logger.error("Error processing document: %s", exc)
        await update.message.reply_text("❌ Sorry, something went wrong while processing your image. Please try again.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fallback for unexpected text messages."""
    await update.message.reply_text(
        "📸 Please send me an image to caption!\nType /help for usage instructions."
    )

# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("about", about_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot is running…")
    app.run_polling()


if __name__ == "__main__":
    main()
