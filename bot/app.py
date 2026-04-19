import os
import httpx
from dotenv import load_dotenv
import discord
from discord import app_commands
from discord.ext import commands

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
API_URL = os.getenv("API_URL", "http://api:8000")

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
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.post(f"{API_URL}/v1/roast", json={"target": name, "invoker": invoker})
            r.raise_for_status()
            data = r.json()
            # Only post if API returned a text field
            text = data.get("text")
            if text:
                await interaction.followup.send(text)
            else:
                # Per rule: bot should not send clarifications or other messages
                # Do nothing
                pass
    except Exception:
        # Fail silently per rule: bot only posts insults. If generation fails, don't post anything.
        pass


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise SystemExit("DISCORD_TOKEN not set in .env")
    bot.run(DISCORD_TOKEN)
