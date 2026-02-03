import logging
import os
import json
import base64
import io
import asyncio
import html
import uuid
import aiohttp
import aiosqlite
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, CallbackQueryHandler, filters

# Configuration
SD_URL = "https://uvgiq-34-138-19-21.a.free.pinggy.link" 
TOKEN = "7942698199:AAEg1z9jqhUp5GnClWepqTH9zyCzRB4ARfw"
DB_NAME = "bot_data.db"

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

async def init_db():
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS generations (
                id TEXT PRIMARY KEY,
                prompt TEXT,
                seed INTEGER
            )
        ''')
        await db.commit()

async def save_generation(req_id, prompt, seed):
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute('INSERT INTO generations (id, prompt, seed) VALUES (?, ?, ?)', (req_id, prompt, seed))
        await db.commit()

async def get_generation(req_id):
    async with aiosqlite.connect(DB_NAME) as db:
        async with db.execute('SELECT prompt, seed FROM generations WHERE id = ?', (req_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return {'prompt': row[0], 'seed': row[1]}
            return None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id, 
        text="Hola! Soy AnaBot. Enviame cualquier texto y generaré imagenes de alta calidad."
    )

def parse_infotext(infotext):
    data = {}
    if not infotext:
        return data

    if "Steps: " in infotext:
        prompt_part, params_part = infotext.split("Steps: ", 1)
        params_part = "Steps: " + params_part
    else:
        prompt_part = infotext
        params_part = ""

    if "Negative prompt: " in prompt_part:
        positive, negative = prompt_part.split("Negative prompt: ", 1)
        data['prompt'] = positive.strip().rstrip(',')
        data['negative_prompt'] = negative.strip().rstrip(',')
    else:
        data['prompt'] = prompt_part.strip().rstrip(',')
        data['negative_prompt'] = ""

    tokens = params_part.split(", ")
    current_key = None
    accumulated_value = []
    
    params = {}
    for token in tokens:
        if ": " in token:
            parts = token.split(": ", 1)
            possible_key = parts[0]
            val = parts[1]
            if current_key:
                params[current_key] = ", ".join(accumulated_value)
            current_key = possible_key
            accumulated_value = [val]
        else:
            if current_key:
                accumulated_value.append(token)
    
    if current_key:
        params[current_key] = ", ".join(accumulated_value)
        
    data.update(params)
    return data

def format_caption(parsed_data, user_name="User"):
    esc = html.escape
    prompt = parsed_data.get('prompt', 'N/A')
    
    config_html = "⚙️ <b>Configuración:</b>\n"
    config_html += f"• <b>Pasos:</b> {esc(str(parsed_data.get('Steps', '?')))}\n"
    config_html += f"• <b>Sampler:</b> {esc(str(parsed_data.get('Sampler', '?')))}\n"
    if 'Schedule type' in parsed_data:
        config_html += f"• <b>Scheduler:</b> {esc(str(parsed_data.get('Schedule type', '?')))}\n"
    config_html += f"• <b>CFG:</b> {esc(str(parsed_data.get('CFG scale', '?')))}\n"
    config_html += f"• <b>Seed:</b> <code>{esc(str(parsed_data.get('Seed', '?')))}</code>\n"
    config_html += f"• <b>Tamaño:</b> {esc(str(parsed_data.get('Size', '?')))}\n"
    
    author_html = f"\n👤 <b>Autor:</b> {esc(user_name)}"
    header_html = "✅ 🎨 <b>Generación completada</b>\n\n"
    
    prompt_html = f"📝 <b>Prompt:</b>\n<code>{esc(prompt)}</code>\n\n"
    
    final_html = header_html + prompt_html + config_html + author_html
    return final_html

async def download_image_to_base64(file_id, context):
    """Downloads image from Telegram and converts to Base64"""
    new_file = await context.bot.get_file(file_id)
    byte_array = await new_file.download_as_bytearray()
    return base64.b64encode(byte_array).decode('utf-8')

async def generate_images(payload):
    """
    Generic generator using a full payload dict.
    """
    url = f"{SD_URL}/sdapi/v1/txt2img"
    timeout = aiohttp.ClientTimeout(total=None)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"SD API Error {response.status}: {text}")
            
            data = await response.json()
            images_b64 = data.get('images', [])
            info_json_str = data.get('info', '{}')
            
            decoded_images = []
            for img_str in images_b64:
                decoded_images.append(base64.b64decode(img_str))
                
            return decoded_images, info_json_str

async def generate_img2img(payload):
    url = f"{SD_URL}/sdapi/v1/img2img"
    timeout = aiohttp.ClientTimeout(total=None)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"SD API Error {response.status}: {text}")
            
            data = await response.json()
            images_b64 = data.get('images', [])
            info_json_str = data.get('info', '{}')
            
            decoded_images = []
            for img_str in images_b64:
                decoded_images.append(base64.b64decode(img_str))
                
            return decoded_images, info_json_str

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_prompt = update.message.text
    user_name = update.effective_user.first_name
    chat_id = update.effective_chat.id
    
    logger.info(f"Received prompt from {user_name}: {user_prompt}")
    
    status_msg = await context.bot.send_message(
        chat_id=chat_id, 
        text=f"🎨 Generando imagenes para: '{user_prompt}'\n⏳ Por favor espera..."
    )

    # Base payload
    payload = {
        "prompt": user_prompt + " ,masterpiece,best quality,amazing quality",
        "negative_prompt": "bad quality,worst quality,worst detail,sketch,censor",
        "steps": 15,
        "cfg_scale": 5,
        "width": 1024,
        "height": 1024,
        "n_iter": 2, 
        "batch_size": 1,
        "sampler_name": "Euler",
        "scheduler": "Normal",
    }

    try:
        images_bytes, info_json_str = await generate_images(payload)
        await process_and_send_images(context, chat_id, images_bytes, info_json_str, user_name, user_prompt, source_action="txt2img")
        await context.bot.delete_message(chat_id=chat_id, message_id=status_msg.message_id)

    except Exception as e:
        logger.error(f"Error generating image: {e}", exc_info=True)
        await context.bot.send_message(chat_id=chat_id, text=f"❌ Error al generar: {str(e)}")

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles Upscale, Repeat and Super Upscale buttons.
    """
    query = update.callback_query
    await query.answer() 
    
    data = query.data # "upscale:<uuid>" | "repeat:<uuid>" | "super_upscale:<uuid>"
    action, cache_id = data.split(":", 1)
    
    cached_data = await get_generation(cache_id)
    if not cached_data:
        await query.edit_message_caption(caption="❌ Datos no encontrados en base de datos. Genera nuevo.")
        return

    chat_id = query.message.chat_id
    user_name = update.effective_user.first_name
    original_prompt = cached_data['prompt'] 
    seed = cached_data['seed']
    
    # Status Message
    status_text = f"⚙️ Procesando {action.upper().replace('_', ' ')}...\n⏳ Por favor espera..."
    status_msg = await context.bot.send_message(
        chat_id=chat_id, 
        text=status_text
    )

    try:
        images_bytes = []
        info_json_str = "{}"
        
        # Determine next button state
        next_source_action = "txt2img" # Default fallback
        
        if action == "upscale":
             # TXT2IMG UPSCALE (1.5x + ADetailer)
             payload = {
                "prompt": original_prompt + " ,masterpiece,best quality,amazing quality",
                "negative_prompt": "bad quality,worst quality,worst detail,sketch,censor",
                "steps": 15,
                "cfg_scale": 5,
                "width": 1024,
                "height": 1024,
                "n_iter": 1, 
                "batch_size": 1,
                "sampler_name": "Euler",
                "scheduler": "Normal",
                "seed": seed,
                "enable_hr": True,
                "denoising_strength": 0.4,
                "hr_scale": 1.5,
                "hr_upscaler": "R-ESRGAN 4x+ Anime6B",
                "hr_second_pass_steps": 20,
                "alwayson_scripts": {
                    "ADetailer": {
                        "args": [
                            {"ad_model": "face_yolov8n.pt"},
                            {"ad_model": "mediapipe_face_full"},
                            {"ad_model": "mediapipe_face_mesh_eyes_only"}
                        ]
                    }
                }
             }
             images_bytes, info_json_str = await generate_images(payload)
             next_source_action = "upscale" # Result of upscale -> shows Super Upscale
             
        elif action == "repeat":
             # NEW VARIATION
             payload = {
                "prompt": original_prompt + " ,masterpiece,best quality,amazing quality",
                "negative_prompt": "bad quality,worst quality,worst detail,sketch,censor",
                "steps": 15,
                "cfg_scale": 5,
                "width": 1024,
                "height": 1024,
                "n_iter": 1,
                "batch_size": 1, 
                "seed": -1 
             }
             images_bytes, info_json_str = await generate_images(payload)
             next_source_action = "txt2img" # Standard buttons
             
        elif action == "super_upscale":
             # ULTIMATE SD UPSCALE
             # Need to download the image from the message that was clicked
             # The message contains the document.
             
             if not query.message.document:
                 await context.bot.send_message(chat_id=chat_id, text="❌ No se encontró documento base para upscale.")
                 return

             base64_image = await download_image_to_base64(query.message.document.file_id, context)
             
             payload = {
                 "init_images": [base64_image],
                 "prompt": "best quality, masterpiece, highres, ultra detailed, 8k wallpaper",
                 "negative_prompt": "blur, low quality, lowres, watermark, text, deformed, bad anatomy",
                 "steps": 30,
                 "sampler_name": "DPM++ 2M Karras",
                 "cfg_scale": 7,
                 "denoising_strength": 0.35, # CRITICAL
                 "seed": -1, # Random seed per requirements (or could use same, but user prompt said -1)
                 "sampler_name": "Euler",
                 "scheduler": "Normal",
                 "script_name": "ultimate sd upscale",
                 "script_args": [
                    None,            # Dummy (README says null)
                    1024,            # Tile width (Updated to 1024)
                    1024,            # Tile height (Updated to 1024)
                    12,              # Mask blur (User screenshot: 12)
                    64,              # Padding (User screenshot: 64)
                    64,              # Seams fix width
                    0.35,            # Seams fix denoise
                    32,              # Seams fix padding
                    4,               # Upscaler (Index 4 confirmed)
                    True,            # Save upscaled image
                    0,               # Redraw mode (Linear)
                    False,           # Save seams fix image
                    8,               # Seams fix mask blur
                    0,               # Seams fix type (None)
                    2,               # Target size type (Scale from image size)
                    2048,            # Custom width
                    2048,            # Custom height
                    2.6              # Scale
                 ],
                 "alwayson_scripts": {
                     "ControlNet": {
                         "args": [
                             {
                                 "module": "tile_resample",
                                 "model": "control_v11f1e_sd15_tile.pth",
                                 "pixel_perfect": True
                             }
                         ]
                     }
                 }
             }
             
             logger.info(f"Sending SUPER UPSCALE payload with {len(payload['script_args'])} args: {payload['script_args']}")
             images_bytes, info_json_str = await generate_img2img(payload)
             next_source_action = "super_upscale" # Result of super -> No buttons

        await process_and_send_images(context, chat_id, images_bytes, info_json_str, user_name, original_prompt, source_action=next_source_action)
        await context.bot.delete_message(chat_id=chat_id, message_id=status_msg.message_id)

    except Exception as e:
        logger.error(f"Error in {action}: {e}", exc_info=True)
        await context.bot.send_message(chat_id=chat_id, text=f"❌ Error al procesar: {str(e)}")

async def process_and_send_images(context, chat_id, images_bytes, info_json_str, user_name, base_prompt, source_action="txt2img"):
    """
    Helper to parse response, format caption, cache data in DB, and send message with buttons.
    source_action determines which buttons to show.
    """
    info_data = json.loads(info_json_str)
    infotexts = info_data.get('infotexts', [])
    all_seeds = info_data.get('all_seeds', [])

    for i, img_data in enumerate(images_bytes):
        current_infotext = infotexts[i] if i < len(infotexts) else ""
        parsed_data = parse_infotext(current_infotext)
        
        # Ensure seed
        current_seed = parsed_data.get('Seed')
        if not current_seed:
             if i < len(all_seeds):
                 current_seed = all_seeds[i]
                 parsed_data['Seed'] = current_seed
        
        # Save to SQLite
        req_id = str(uuid.uuid4())
        # We always save it, so even super upscaled images *could* be retrieved if we added buttons later
        await save_generation(req_id, base_prompt, int(current_seed) if current_seed else -1)
        
        # Button Logic
        reply_markup = None
        
        if source_action == "txt2img":
            # Initial Generaton: [Upscale] [Repeat]
            keyboard = [[
                InlineKeyboardButton("✨ UPSCALE", callback_data=f"upscale:{req_id}"),
                InlineKeyboardButton("🔄 REPETIR", callback_data=f"repeat:{req_id}")
            ]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
        elif source_action == "upscale":
            # Result of Upscale: [SUPER UPSCALE] only
            keyboard = [[
                InlineKeyboardButton("🚀 SUPER UPSCALE", callback_data=f"super_upscale:{req_id}")
            ]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
        elif source_action == "super_upscale":
            # Result of Super Upscale: No buttons (or maybe simple repeat?)
            # User said "debe tener uno solo" for the *upscale result* which leads here.
            # For this final image, we assume no further definition.
            reply_markup = None

        full_caption_html = format_caption(parsed_data, user_name)

        if len(full_caption_html) > 1024:
            await context.bot.send_document(
                chat_id=chat_id,
                document=io.BytesIO(img_data),
                filename=f"image_{uuid.uuid4().hex[:8]}.png",
                caption="✅ 🎨 <b>Generación completada</b>\n(Ver detalles abajo 👇)",
                parse_mode=ParseMode.HTML,
                reply_markup=reply_markup
            )
            await context.bot.send_message(
                chat_id=chat_id,
                text=full_caption_html,
                parse_mode=ParseMode.HTML
            )
        else:
            await context.bot.send_document(
                chat_id=chat_id,
                document=io.BytesIO(img_data),
                filename=f"image_{uuid.uuid4().hex[:8]}.png",
                caption=full_caption_html,
                parse_mode=ParseMode.HTML,
                reply_markup=reply_markup
            )

if __name__ == '__main__':
    async def post_init(application):
        await init_db()
        logger.info("Database initialized.")

    application = ApplicationBuilder().token(TOKEN).post_init(post_init).build()
    
    start_handler = CommandHandler('start', start)
    message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    callback_handler = CallbackQueryHandler(button_handler)
    
    application.add_handler(start_handler)
    application.add_handler(message_handler)
    application.add_handler(callback_handler)
    
    print("Bot started with SUPER UPSCALE. Press Ctrl+C to stop.")
    application.run_polling()
