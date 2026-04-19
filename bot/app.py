import os
import httpx
from dotenv import load_dotenv
import discord # pyright: ignore[reportMissingImports]
from discord import app_commands
from discord.ext import commands

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
API_URL = os.getenv("API_URL", "http://api:8000")
BOT_API_TIMEOUT = float(os.getenv("BOT_API_TIMEOUT", "45"))

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (id: {bot.user.id})")
    try:
        await bot.tree.sync()
    except Exception:
        pass


@bot.tree.command(name="insultar", description="Insulta con estilo roast comediante")
@app_commands.describe(target="Usuario a insultar")
async def insult(interaction: discord.Interaction, target: discord.Member):
    await interaction.response.defer()
    name = target.display_name or target.name
    invoker = interaction.user.display_name or interaction.user.name
    fallback_text = f"{name}, sos tan desastre que hasta el silencio te bardea solo."
    try:
        async with httpx.AsyncClient(timeout=BOT_API_TIMEOUT) as client:
            r = await client.post(f"{API_URL}/v1/roast", json={"target": name, "invoker": invoker})
            r.raise_for_status()
            data = r.json()
            text = data.get("text")
            if text:
                await interaction.followup.send(text)
            else:
                await interaction.followup.send(fallback_text)
    except Exception as e:
        print(f"Error calling API: {type(e).__name__}: {e}")
        try:
            await interaction.followup.send(fallback_text)
        except Exception as followup_error:
            print(f"Error sending followup: {type(followup_error).__name__}: {followup_error}")


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise SystemExit("DISCORD_TOKEN not set in .env")
    bot.run(DISCORD_TOKEN)
