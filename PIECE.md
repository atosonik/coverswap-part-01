# coverswap-part-01

Piece 1 of 9 from https://github.com/atosonik/coverswap.git.

This repo is one new commit. Paths match the original tree. History was not copied.
File bytes in this piece: 374.5 MB. Files: 413.

## Put the tree back together

```bash
mkdir -p coverswap-pieces && cd coverswap-pieces
git clone --depth 1 https://github.com/atosonik/coverswap-part-01.git
git clone --depth 1 https://github.com/atosonik/coverswap-part-02.git
git clone --depth 1 https://github.com/atosonik/coverswap-part-03.git
git clone --depth 1 https://github.com/atosonik/coverswap-part-04.git
git clone --depth 1 https://github.com/atosonik/coverswap-part-05.git
git clone --depth 1 https://github.com/atosonik/coverswap-part-06.git
git clone --depth 1 https://github.com/atosonik/coverswap-part-07.git
git clone --depth 1 https://github.com/atosonik/coverswap-part-08.git
git clone --depth 1 https://github.com/atosonik/coverswap-part-09.git
cd ..
mkdir -p coverswap
for d in coverswap-pieces/*/; do
  tar -C "$d" --exclude .git --exclude PIECE.md --exclude SPLIT.json -cf - . | tar -C coverswap -xf -
done
cd coverswap
python chunker.py merge
```

`chunker.py merge` rebuilds `weights/` from `parts/`. Every piece is listed in `SPLIT.json`.
