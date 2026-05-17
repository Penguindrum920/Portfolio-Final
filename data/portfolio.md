# Portfolio OS Knowledge

## Main Concept
Aditya's portfolio simulates a Windows-like desktop environment inside a web page. It is not a normal landing page. The first experience is an interactive desktop with wallpaper, icons, taskbar, Start menu, app windows, and system-like utilities.

## Main Entry
- `index.html` redirects to `code.html`.
- `code.html` is the main portfolio UI.
- The React app in `frontend/` is a smaller prototype for the Aditya AI assistant/action contract, while `code.html` is the canonical shipped portfolio interface.

## Desktop Features
- Desktop icons for About Me, Projects, Skills, Socials, Store, Recycle Bin, Aditya AI, and Games.
- File Explorer windows with folders and item views.
- Browser window called Chrome, with an address bar and embedded previews.
- Windows Terminal with supported commands like `open about`, `open projects`, `open games`, `open skills`, `open socials`, `open store`, `open recyclebin`, `open chrome`, `open terminal`, `open notepad`, `open adityaai`, `theme dark`, `theme light`, `time`, `date`, `clear`, and `shutdown`.
- Notepad with a welcome note that explains the chatbot is personalized and agentic.
- Games Launcher with Snake and Minesweeper.
- Microsoft Store-style app that installs project shortcuts for Parkinson Assessment, AniVerse, and Pegasus.
- Recycle Bin behavior for deleted installed project apps, including restore and empty actions.
- Wallpaper personalization with multiple Windows-style wallpapers.
- Calendar and clock flyout.
- Light and dark theme support, with dark mode intended as the default.

## Aditya AI Behavior
Aditya AI is the embedded assistant. It should answer questions only from Aditya's portfolio knowledge, including about Aditya, his skills, projects, contact links, and portfolio UI. It should not answer unrelated general knowledge. If information is not in the portfolio knowledge, it should say: I don't have that information yet.

Aditya AI also supports action commands. Examples:
- `open projects` opens the Projects folder.
- `open resume` opens the resume PDF.
- `open skills` opens the Skills folder.
- `open socials` or `open contact` opens the Socials folder.
- `play snake` opens Snake.
- `play minesweeper` opens Minesweeper.
- `open games` opens the Games Launcher.

## Project Showcase
The project showcase is opened through `Projects/demo.html`. It presents AniVerse, Parkinson Disease Assessment Portal, and Pegasus with animated sections, screenshot galleries, cursor trails, and an Exit control that returns to the desktop overlay.
