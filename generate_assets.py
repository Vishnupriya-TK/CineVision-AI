"""
Generate logo and banner assets for CineVision AI.
Run once: python generate_assets.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def create_logo():
    """Create a Netflix-inspired logo."""
    size = (400, 400)
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background circle
    draw.ellipse([20, 20, 380, 380], fill=(229, 9, 20, 255))

    # Film reel icon (simplified)
    draw.ellipse([120, 120, 280, 280], outline=(255, 255, 255, 255), width=8)
    draw.ellipse([155, 155, 245, 245], fill=(229, 9, 20, 255))
    # Spokes
    for angle in range(0, 360, 45):
        import math
        cx, cy = 200, 200
        r1, r2 = 30, 55
        x1 = cx + r1 * math.cos(math.radians(angle))
        y1 = cy + r1 * math.sin(math.radians(angle))
        x2 = cx + r2 * math.cos(math.radians(angle))
        y2 = cy + r2 * math.sin(math.radians(angle))
        draw.line([(x1, y1), (x2, y2)], fill=(255, 255, 255, 200), width=3)

    # Text
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    draw.text((200, 310), "CineVision", fill=(255, 255, 255, 255), anchor="mm", font=font)

    img.save(ASSETS_DIR / "logo.png")
    print(f"Created {ASSETS_DIR / 'logo.png'}")


def create_banner():
    """Create a cinematic hero banner."""
    width, height = 1200, 350
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    # Gradient background
    for y in range(height):
        r = int(10 + (y / height) * 20)
        g = int(10 + (y / height) * 15)
        b = int(20 + (y / height) * 40)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Red accent stripe
    draw.rectangle([0, height - 6, width, height], fill=(229, 9, 20))

    # Decorative film strips
    for x in range(0, width, 80):
        draw.rectangle([x, 30, x + 40, 60], fill=(229, 9, 20, 180))
        draw.rectangle([x + 20, height - 80, x + 60, height - 50], fill=(255, 215, 0, 120))

    try:
        title_font = ImageFont.truetype("arial.ttf", 52)
        sub_font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()

    draw.text((600, 140), "CineVision AI", fill=(255, 255, 255), anchor="mm", font=title_font)
    draw.text(
        (600, 200),
        "Movie Success Predictor — Powered by Machine Learning",
        fill=(200, 200, 200),
        anchor="mm",
        font=sub_font,
    )

    img.save(ASSETS_DIR / "banner.jpg", quality=95)
    print(f"Created {ASSETS_DIR / 'banner.jpg'}")


if __name__ == "__main__":
    create_logo()
    create_banner()
    print("Assets generated successfully!")
