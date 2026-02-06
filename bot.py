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
import random
import re
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, CallbackQueryHandler, filters

# Configuration
SD_URL = "http://localhost:7860"

TOKEN = "7942698199:AAEg1z9jqhUp5GnClWepqTH9zyCzRB4ARfw"
DB_NAME = "bot_data.db"

BASE_PROMPT_PREFIX = ""
BASE_PROMPT_SUFFIX = ",masterpiece,best quality,amazing quality"
BASE_NEGATIVE_PROMPT = "bad quality,worst quality,worst detail,sketch,censor"
BASE_WIDTH = 832
BASE_HEIGHT = 1216
BASE_STEPS = 8
BASE_CFG_SCALE = 1.0
BASE_ITER = 1
BASE_SAMPLER = "Euler a"
BASE_SCHEDULER = "Normal"

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

async def init_db():
    async with aiosqlite.connect(DB_NAME) as db:
        # Initial creation for fresh install (old schema support)
        await db.execute('''
            CREATE TABLE IF NOT EXISTS generations (
                id TEXT PRIMARY KEY,
                prompt TEXT,
                seed INTEGER
            )
        ''')
        
        # Migrations
        try:
            await db.execute(f"ALTER TABLE generations ADD COLUMN steps INTEGER DEFAULT {BASE_STEPS}")
            logger.info("Column 'steps' added.")
        except Exception:
            pass
        try:
            await db.execute(f"ALTER TABLE generations ADD COLUMN cfg_scale REAL DEFAULT {BASE_CFG_SCALE}")
            logger.info("Column 'cfg_scale' added.")
        except Exception:
            pass
        try:
            await db.execute(f"ALTER TABLE generations ADD COLUMN sampler_name TEXT DEFAULT '{BASE_SAMPLER}'")
            logger.info("Column 'sampler_name' added.")
        except Exception:
            pass
        try:
            await db.execute(f"ALTER TABLE generations ADD COLUMN scheduler TEXT DEFAULT '{BASE_SCHEDULER}'")
            logger.info("Column 'scheduler' added.")
        except Exception:
            pass

        # Ensure correct schema for future
        await db.execute('''
            CREATE TABLE IF NOT EXISTS generations (
                id TEXT PRIMARY KEY,
                prompt TEXT,
                seed INTEGER,
                steps INTEGER,
                cfg_scale REAL,
                sampler_name TEXT,
                scheduler TEXT
            )
        ''')
        await db.commit()

async def save_generation(req_id, prompt, seed, steps, cfg_scale, sampler_name, scheduler):
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute(
            'INSERT INTO generations (id, prompt, seed, steps, cfg_scale, sampler_name, scheduler) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (req_id, prompt, seed, steps, cfg_scale, sampler_name, scheduler)
        )
        await db.commit()

async def get_generation(req_id):
    async with aiosqlite.connect(DB_NAME) as db:
        async with db.execute('SELECT prompt, seed, steps, cfg_scale, sampler_name, scheduler FROM generations WHERE id = ?', (req_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return {
                    'prompt': row[0], 
                    'seed': row[1],
                    'steps': row[2],
                    'cfg_scale': row[3],
                    'sampler_name': row[4],
                    'scheduler': row[5]
                }
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

def process_dynamic_keywords(prompt):
    """
    Scans resources folder for txt files.
    If filename matches a word in prompt using regex word boundary, replaces it with {line|line|...}
    using 5-10 random lines.
    Uses re.sub to ensure each occurrence gets a FRESH random sample.
    """
    resources_dir = "resources"
    if not os.path.exists(resources_dir):
        return prompt
        
    for filename in os.listdir(resources_dir):
        if filename.endswith(".txt"):
            keyword = os.path.splitext(filename)[0]
            
            # Regex to find keyword as a whole word (so f_anime doesn't match f_anime_2)
            # Escaping keyword just in case it has special chars
            pattern = r'\b' + re.escape(keyword) + r'\b'
            
            if re.search(pattern, prompt):
                file_path = os.path.join(resources_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        # Handle case where file might be one big line or mixed newlines
                        # Also replace | with , to avoid breaking syntax inside options
                        raw_lines = content.replace('\r\n', '\n').replace('\r', '\n').split('\n')
                        # Sanitize and unique
                        lines = list(set(line.strip().replace('|', ',') for line in raw_lines if line.strip()))
                    
                    if not lines:
                        continue
                        
                    def replace_callback(match):
                        # Pick 10 to 15 lines (or all if fewer)
                        count = min(len(lines), random.randint(10, 15))
                        selected = random.sample(lines, count)
                        # Construct dynamic prompt syntax: {a|b|c}
                        return "{" + "|".join(selected) + "}"
                    
                    # Replace in prompt
                    new_prompt = re.sub(pattern, replace_callback, prompt)
                    
                    if new_prompt != prompt:
                         logger.info(f"Replaced keyword '{keyword}' with {len(lines)} available options.")
                         prompt = new_prompt
                    
                except Exception as e:
                    logger.error(f"Error processing resource {filename}: {e}")
                    
    return prompt

async def generate_images(payload):
    """
    Generic generator using a full payload dict.
    """
    url = f"{SD_URL}/sdapi/v1/txt2img"
    timeout = aiohttp.ClientTimeout(total=None)

    # DEBUG: Save payload
    with open("debug_payload.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved debug_payload.json")

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

async def generate_extras_upscale(payload):
    url = f"{SD_URL}/sdapi/v1/extra-single-image"
    timeout = aiohttp.ClientTimeout(total=None)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"SD API Error {response.status}: {text}")
            
            text = await response.text()
            if not text:
                raise Exception("API returned empty response")
                
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                raise Exception(f"Failed to decode JSON. Response text: {text[:200]}...") # Truncate for safety

            image_b64 = data.get('image', "")
            info_json_str = "{}" # data.get('html_info', '{}') returns HTML which crashes json.loads later
            
            # extras returns a single image
            decoded_images = []
            if image_b64:
                 decoded_images.append(base64.b64decode(image_b64))
                
            return decoded_images, info_json_str

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_prompt = update.message.text
    user_name = update.effective_user.first_name
    chat_id = update.effective_chat.id
    
    logger.info(f"Received prompt from {user_name}: {user_prompt}")
    
    # Process Dynamic Keywords
    processed_prompt = process_dynamic_keywords(user_prompt)
    if processed_prompt != user_prompt:
        logger.info(f"Expanded Prompt: {processed_prompt}")
        # Note: We don't overwrite user_prompt for the "Generating..." message 
        # so the user sees what they typed, but we use processed_prompt for generation.
    
    status_msg = await context.bot.send_message(
        chat_id=chat_id, 
        text=f"🎨 Generando imagenes para: '{user_prompt}'\n⏳ Por favor espera..."
    )

    # Base payload
    payload = {
        "prompt": BASE_PROMPT_PREFIX + processed_prompt + BASE_PROMPT_SUFFIX,
        "negative_prompt": BASE_NEGATIVE_PROMPT,
        "steps": BASE_STEPS,
        "cfg_scale": BASE_CFG_SCALE,
        "width": BASE_WIDTH,
        "height": BASE_HEIGHT,
        "n_iter": BASE_ITER, 
        "batch_size": 1,
        "sampler_name": BASE_SAMPLER,
        "scheduler": BASE_SCHEDULER,
    }

    try:
        images_bytes, info_json_str = await generate_images(payload)
        await process_and_send_images(context, chat_id, images_bytes, info_json_str, user_name, processed_prompt, source_action="txt2img")
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
    steps = cached_data.get('steps', BASE_STEPS)
    cfg_scale = cached_data.get('cfg_scale', BASE_CFG_SCALE)
    sampler_name = cached_data.get('sampler_name', BASE_SAMPLER)
    scheduler = cached_data.get('scheduler', BASE_SCHEDULER)
    
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
              # TXT2IMG UPSCALE (1.5x)
              payload = {
                "prompt": original_prompt, # Prompt from DB already has prefix/suffix
                "negative_prompt": BASE_NEGATIVE_PROMPT,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "width": BASE_WIDTH,
                "height": BASE_HEIGHT,
                "n_iter": 1, 
                "batch_size": 1,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "seed": seed,
                "enable_hr": True,
                "denoising_strength": 0.4,
                "hr_scale": 1.5,
                "hr_upscaler": "R-ESRGAN 4x+ Anime6B",
                "hr_sampler_name": sampler_name,  # Match original
                "hr_scheduler": scheduler,        # Match original
                "hr_second_pass_steps": 20,
                "alwayson_scripts": {} # Clear ADetailer for parity
              }
              images_bytes, info_json_str = await generate_images(payload)
              next_source_action = "upscale" # Result of upscale -> shows Super Upscale
             
        elif action == "repeat":
             # NEW VARIATION (Same expanded prompt, new seed)
             payload = {
                "prompt": original_prompt, 
                "negative_prompt": BASE_NEGATIVE_PROMPT,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "width": BASE_WIDTH,
                "height": BASE_HEIGHT,
                "n_iter": 1,
                "batch_size": 1, 
                "sampler_name": sampler_name,
                "scheduler": scheduler,
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
                 "prompt": original_prompt, # Use original expanded prompt
                 "negative_prompt": BASE_NEGATIVE_PROMPT,
                 "steps": steps,
                 "sampler_name": sampler_name,
                 "cfg_scale": cfg_scale,
                 "denoising_strength": 0.35,
                 "seed": seed, # Use original seed for consistency
                 "scheduler": scheduler,
                 "script_name": "ultimate sd upscale",
                 "script_args": [
                    None,
                    BASE_WIDTH,
                    BASE_HEIGHT,
                    12,
                    64,
                    64,
                    0.35,
                    32,
                    4, # Upscaler index
                    True,
                    0,
                    False,
                    8,
                    0,
                    2, # Scale from image size
                    2048,
                    2048,
                    2.0 # Standard 2x scale
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

        elif action == "fast_upscale":
             # EXTRAS SINGLE IMAGE UPSCALE
             if not query.message.document:
                 await context.bot.send_message(chat_id=chat_id, text="❌ No se encontró documento base para upscale.")
                 return

             base64_image = await download_image_to_base64(query.message.document.file_id, context)
             
             payload = {
                "resize_mode": 0,
                "show_extras_results": True,
                "gfpgan_visibility": 0,
                "codeformer_visibility": 0,
                "codeformer_weight": 0,
                "upscaling_resize": 3,
                "upscaling_resize_w": 512,
                "upscaling_resize_h": 512,
                "upscaling_crop": True,
                "upscaler_1": "R-ESRGAN 4x+ Anime6B",
                "upscaler_2": "None",
                "extras_upscaler_2_visibility": 0,
                "upscale_first": False,
                "image": base64_image
             }
             
             # For extras, info string might be different or empty, we handle it generic
             images_bytes, info_json_str = await generate_extras_upscale(payload)
             
             # RECONSTRUCT INFOTEXT FROM DB CACHE
             # Since extras doesn't return generation params, we fake it so the caption looks good.
             # Format: "Prompt\nSteps: XX, Sampler: YY, CFG scale: ZZ, Seed: SS, Size: WxH"
             
             synthetic_infotext = f"{original_prompt}\nSteps: {steps}, Sampler: {sampler_name}, CFG scale: {cfg_scale}, Seed: {seed}, Size: {BASE_WIDTH}x{BASE_HEIGHT}"
             # Note: Size is hardcoded relative to base, but fast upscale is x3. 
             # Maybe we show the *Target* size? 1024*3 = 3072. Let's start with base or maybe generic.
             
             synthetic_info = {
                 "infotexts": [synthetic_infotext],
                 "all_seeds": [seed]
             }
             info_json_str = json.dumps(synthetic_info)
             
             next_source_action = "fast_upscale" # Result of fast upscale -> No buttons

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
        
        # EXTRACT METADATA for saving
        # parsed_data is from infotext, which is most reliable for what actually happened
        actual_prompt = parsed_data.get('prompt', base_prompt) # This is the key fix: save expanded prompt
        meta_steps = int(parsed_data.get('Steps', BASE_STEPS))
        meta_cfg = float(parsed_data.get('CFG scale', BASE_CFG_SCALE))
        meta_sampler = parsed_data.get('Sampler', BASE_SAMPLER)
        meta_scheduler = parsed_data.get('Schedule type', BASE_SCHEDULER)

        # We always save it, so even super upscaled images *could* be retrieved if we added buttons later
        await save_generation(req_id, actual_prompt, int(current_seed) if current_seed else -1, meta_steps, meta_cfg, meta_sampler, meta_scheduler)
        
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
            # Result of Upscale: [SUPER UPSCALE] [FAST UPSCALE]
            keyboard = [[
                InlineKeyboardButton("🚀 SUPER UPSCALE", callback_data=f"super_upscale:{req_id}"),
                InlineKeyboardButton("⚡ FAST UPSCALE", callback_data=f"fast_upscale:{req_id}")
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
