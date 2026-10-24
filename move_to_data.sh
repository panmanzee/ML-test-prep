#!/usr/bin/env bash
# Move big folders from the full root partition (/) to /mnt/data and leave a symlink behind.
# Asks before every folder. Programs keep working because the old path still points to the files.
set -u

DEST=/mnt/data/pan/home
CANDIDATES=(
  anaconda3 .conda .Xilinx .Xil STM32CubeIDE STM32Cube .stm32cubeide .stm32cubemx .stmcube
  .cache .npm .npm-global .nuget .dotnet .vscode .eclipse
  Downloads Videos Pictures Music Documents
)

echo "== Before =="
df -h / /mnt/data | sed 1d
echo

# 1) Remove old (disabled) snap revisions
read -r -p "Remove old snap revisions first? (y/N) " a
if [[ $a == [yY] ]]; then
  snap list --all | awk '/disabled/{print $1, $3}' | while read -r name rev; do
    echo "  removing snap $name rev $rev"
    sudo snap remove "$name" --revision="$rev"
  done
  sudo snap set system refresh.retain=2
fi
echo

sudo mkdir -p "$DEST" && sudo chown "$USER:$USER" /mnt/data/pan "$DEST"

# 2) Show sizes, biggest first
echo "Measuring folders (takes a moment)..."
mapfile -t LIST < <(
  for d in "${CANDIDATES[@]}"; do
    p="$HOME/$d"
    [[ -d $p && ! -L $p ]] && du -sxk "$p" 2>/dev/null
  done | sort -rn
)
echo
for line in "${LIST[@]}"; do
  kb=${line%%$'\t'*}; p=${line#*$'\t'}
  printf "  %8s  %s\n" "$(numfmt --to=iec --from-unit=1024 "$kb")" "$p"
done
echo
echo "Close VS Code, Jupyter, Vivado, STM32CubeIDE and any terminal using conda before continuing."
echo

# 3) Move each folder after confirmation
for line in "${LIST[@]}"; do
  p=${line#*$'\t'}; name=$(basename "$p")
  size=$(numfmt --to=iec --from-unit=1024 "${line%%$'\t'*}")
  read -r -p "Move $p ($size) to $DEST/$name ? (y/N) " a
  [[ $a == [yY] ]] || continue
  if [[ -e $DEST/$name ]]; then echo "  $DEST/$name already exists, skipping"; continue; fi
  if rsync -aHAX --info=progress2 "$p/" "$DEST/$name/"; then
    rm -rf "$p" && ln -s "$DEST/$name" "$p" && echo "  done: $p -> $DEST/$name"
  else
    echo "  copy FAILED, original left untouched"; rm -rf "${DEST:?}/$name"
  fi
  df -h / | sed 1d
done

echo
echo "== After =="
df -h / /mnt/data | sed 1d
