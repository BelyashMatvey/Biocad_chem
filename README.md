# Biocad Tool

### Molecular parametrization pipeline for GROMACS using OpenFF

This tool converts ligand structures from an **SDF file** into:

- `.gro` structures (GROMACS format)
- `.top` topology files (OpenFF SMIRNOFF-based)

It supports multiple conformers and parallel processing.

## Input

- SDF file containing ligand structures with 3D coordinates

Example:
```
file.sdf
```

---

## Output

For each unique molecule:

```
outdir/
  MOLECULE_NAME/
    topol.top
    conf_0.gro
    conf_1.gro
    ...
```

---

## Usage

### Command line

```bash
python biocad_tool.py --sdf file.sdf --out outdir --nproc 4 --log run.log
```

---

### Arguments

| Argument | Description |
|----------|------------|
| `--sdf` | Path to input SDF file |
| `--out` | Output directory |
| `--nproc` | Number of CPU cores |
| `--log` | Log file path |


---

## Requirements

#### All requirements are in environment.yaml

---

## Author

**Matvei Beliakov**

---

## P.S.

> Шопинг модный лук\
> Я звезда YouTube

![](pict.png)