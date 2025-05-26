# Oscana: Neutrino Oscillation Analysis Package

## Changelog (v1.0.8)

- Event images can be filled with any appropriate variable in the SNTP files, such as the photoelectron counts (from east or west strip-ends). Each fill is added as a seperate "colour" channel. See `create_fd_full_image` and `create_fd_split_image` functions ("image.py").
- Implemented the `apply_transforms` method to the `DataHandler` class. Now, applied data transformations and cuts can be tracked by the data handler.
- Re-implemented the logic for the `TransformMetadata` and added a `TransformBase` to be used as a template for implementing data transformations and cuts.
- The variable search tool functions are now consolidated in the `VariableSearchTool` (Singleton) class.
- Minor changes to documentation.
