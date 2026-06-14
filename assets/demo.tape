# VHS tape for the README demo GIF.  Regenerate with:  vhs assets/demo.tape
# (vhs: https://github.com/charmbracelet/vhs)

Output assets/demo.gif

Set Shell bash
Set Theme "Catppuccin Mocha"
Set FontSize 18
Set Width 1080
Set Height 640
Set Padding 24
Set TypingSpeed 55ms
Set PlaybackSpeed 1.0

# --- setup (hidden): load the demo output function + a clean prompt ---
Hide
Type "source assets/demo.sh"
Enter
Type "export PS1='\033[36m❯\033[0m '"
Enter
Type "clear"
Enter
Show

# --- the recorded session ---
Sleep 700ms
Type "maestra --csv data/titanic.csv --target Survived"
Sleep 600ms
Enter
Sleep 9s
Sleep 2s
