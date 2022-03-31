# dtl++

This repository contains two Python packages:

- `dtlc`: A compiler for DTL to a number of different backends (e.g. Python, loopy).
- `dtlpp`: A set of extensions to DTL that do not fit within the minimal set found in the language definition (e.g. `UnitVectorSpace` to represent permutation tensors). Implementation specific extensions can be found inside the `dtlpp/impls` subdirectory (e.g. `dtlpp/impls/{numpy,petsc}`).
