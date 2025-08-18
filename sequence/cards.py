"""Card image generation.

This module provides helper functions to generate simple playing card images
for use in the Sequence game UI.  The actual artwork of the commercial
Sequence board cannot be redistributed, so we procedurally render cards
using their rank and suit symbols with Pillow.  Images are cached to
``assets/cards`` to avoid recomputation.  If the directory does not
exist it will be created automatically.

Each card is drawn on a white background with a black border, the rank in
the top-left and bottom-right corners, and the suit symbol centred.  Red
suits (hearts and diamonds) are rendered in red, black suits (clubs and
spades) in black.  Jacks are not present on the board but appear in
players' hands; we draw one-eyed jacks (J1) and two-eyed jacks (J2) with
simple stylised letters to differentiate them.

The corner 'W' squares on the Sequence board are treated as wild and
occupied; for completeness this module can also generate a blank tile
image labelled "W" if required, though the UI may render corners
differently (e.g. highlight them as wild).
"""

from __future__ import annotations

import os
from typing import Dict

from PIL import Image, ImageDraw, ImageFont

# Directory to store generated card images relative to project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARDS_DIR = os.path.join(ROOT_DIR, "assets", "cards")

# Predefine suit symbols using Unicode glyphs.  These are widely
# supported in most fonts.  If Pillow cannot find a font with these
# glyphs, fallback replacements will be used.
SUIT_SYMBOLS: Dict[str, str] = {
    "S": "♠",  # spades
    "H": "♥",  # hearts
    "D": "♦",  # diamonds
    "C": "♣",  # clubs
}

# Basic card dimensions.  Keeping cards modestly sized ensures the board
# fits comfortably in the Tkinter window.  Modify as needed for higher
# resolution.
CARD_WIDTH = 80
CARD_HEIGHT = 110

def _ensure_cards_dir() -> None:
    """Create the card image directory if it doesn't exist."""
    if not os.path.exists(CARDS_DIR):
        os.makedirs(CARDS_DIR, exist_ok=True)

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Attempt to load a default TrueType font; fallback to a basic font.

    Pillow includes a handful of fonts in PIL/fonts; if none are found
    an internal bitmap font will be used (limited glyph support but
    sufficient for numbers and basic letters).
    """
    try:
        # Use DejaVuSans as a reliable default if available
        font_path = ImageFont.truetype("DejaVuSans.ttf", size)
        return font_path
    except Exception:
        # Fallback to default font
        return ImageFont.load_default()

def generate_card_image(card: str) -> Image.Image:
    """Generate and return a PIL Image for the given card code.

    :param card: A two- or three-character code (e.g. '7H', '10S', 'J1', 'J2', 'W').
    :returns: A PIL Image object representing the card.

    Card codes:
        - 'W': wild corner tile
        - 'J1': one-eyed jack (removal)
        - 'J2': two-eyed jack (wild placement)
        - For standard ranks and suits: rank followed by suit letter.  'T' is
          used for ten, but we render '10' on the card face.
    """
    _ensure_cards_dir()
    # Normalise rank and suit from card code
    if card == 'W':
        filename = f"{card}.png"
    else:
        filename = f"{card}.png"
    path = os.path.join(CARDS_DIR, filename)
    if os.path.exists(path):
        return Image.open(path)
    # Create a new blank card image
    image = Image.new("RGBA", (CARD_WIDTH, CARD_HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    # Draw border
    draw.rectangle([(0, 0), (CARD_WIDTH - 1, CARD_HEIGHT - 1)], outline="black", width=2)
    # Determine content based on card type
    font_large = _load_font(40)
    font_small = _load_font(20)
    if card == 'W':
        # Wild corner tile: draw a large "W" centred
        w, h = draw.textsize("W", font=font_large)
        draw.text(((CARD_WIDTH - w) / 2, (CARD_HEIGHT - h) / 2), "W", fill="black", font=font_large)
    elif card in ('J1', 'J2'):
        # Draw J with 1 or 2 eyes (simple stylisation)
        rank_text = "J" + ("₁" if card == 'J1' else "₂")
        w, h = draw.textsize(rank_text, font=font_large)
        draw.text(((CARD_WIDTH - w) / 2, (CARD_HEIGHT - h) / 2), rank_text, fill="black", font=font_large)
        # Label the Jack type at bottom
        label = "One-Eyed" if card == 'J1' else "Two-Eyed"
        lw, lh = draw.textsize(label, font=font_small)
        draw.text(((CARD_WIDTH - lw) / 2, CARD_HEIGHT - lh - 5), label, fill="black", font=font_small)
    else:
        # Split rank and suit
        if card.startswith('10') or (len(card) == 3 and card[:2] == '10'):
            rank = '10'
            suit = card[-1]
        else:
            rank = card[:-1]
            suit = card[-1]
            # Convert 'T' to '10' for display
            if rank == 'T':
                rank = '10'
        suit_symbol = SUIT_SYMBOLS.get(suit, '?')
        # Determine colour: red for hearts/diamonds, black otherwise
        colour = "red" if suit in ('H', 'D') else "black"
        # Draw rank in top-left and bottom-right
        rank_font = _load_font(24)
        rw, rh = draw.textsize(rank, font=rank_font)
        draw.text((4, 2), rank, fill=colour, font=rank_font)
        # Draw the same rank mirrored in bottom-right
        draw.text((CARD_WIDTH - rw - 4, CARD_HEIGHT - rh - 2), rank, fill=colour, font=rank_font)
        # Draw suit symbol centred
        symbol_font = _load_font(48)
        sw, sh = draw.textsize(suit_symbol, font=symbol_font)
        draw.text(((CARD_WIDTH - sw) / 2, (CARD_HEIGHT - sh) / 2 - 10), suit_symbol, fill=colour, font=symbol_font)
    # Save generated image for caching
    image.save(path)
    return image

def generate_all_cards() -> None:
    """Generate and cache images for all cards used in the game.

    This includes standard ranks A–K for each suit, 'J1', 'J2' and the wild
    'W'.  The Ten is represented by 'T' in card codes but will be rendered
    as '10'.
    """
    # Standard ranks used in the deck (excluding Jack codes; handled separately)
    ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'Q', 'K']
    suits = ['S', 'H', 'D', 'C']
    # Generate each rank/suit combination
    for r in ranks:
        for s in suits:
            generate_card_image(f"{r}{s}")
    # Generate jacks and wild tile
    generate_card_image('J1')
    generate_card_image('J2')
    generate_card_image('W')

if __name__ == "__main__":
    # Simple test to generate all card images
    generate_all_cards()