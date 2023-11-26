# cutrace JSON Schema

# Basic Structure
A valid JSON scene contains the following four keys (in any order):
```json
{
  "camera": {},
  "lights": [],
  "materials": [],
  "objects": []
}
```

Each of these is mandatory. The `camera` key should have an object as value, while the others should have an array as value.

# Vector Type
In order to specify 3D-vectors, we use arrays of three elements (to keep it easy to read). The following JSON value corresponds to a vector `{ .x = 0.25, .y = 0.33, .z = 0.5 }`:
```json
[ 0.25, 0.33, 0.5 ]
```

These can be used as point-vectors, direction-vectors or colors (in order RGB).
If a vector is a color, then its components should be between 0 and 1 (both inclusive).

# Camera
The camera has 7 required values (even though some of them aren't used in the code):

 | Key          | Expected type | What                                                  |
 |--------------|---------------|-------------------------------------------------------|
 | `near_plane` | number        | The distance to the near plane of the camera (unused) |
 | `far_plane`  | number        | The distance to the far plane of the camera (unused)  |
 | `eye`        | vector        | The position of the camera                            |
 | `up`         | vector        | The direction that is "up" for the camera             |
 | `look`       | vector        | The point the camera is looking at                    |
 | `width`      | number        | The width of the image to render                      |
 | `height`     | number        | The height of the image to render                     |

# Lights
Currently, `cutrace` supports two kinds of lights: directional (sun) lights, and point lights.
You can disambiguate between the two kinds by using the `type` key (`"sun"` for directional lights, `"point"` for point lights).

## Directional lights
The directional light has three mandatory arguments:

 | Key         | Expected type    | What                                       |
 |-------------|------------------|--------------------------------------------|
 | `type`      | string (`"sun"`) | The type of this light (should be `"sun"`) |
 | `direction` | vector           | The direction this light is shining in     |
 | `color`     | vector           | The color of the light                     |

## Point lights
The point light has three mandatory arguments:

 | Key        | Expected type      | What                                         |
 |------------|--------------------|----------------------------------------------|
 | `type`     | string (`"point"`) | The type of this light (should be `"point"`) |
 | `position` | vector             | The position of this light                   |
 | `color`    | vector             | The color of the light                       |

# Objects
There are four kinds of objects: models (loaded from a file), planes (infinite, flat planes), triangles (a single triangle), and spheres.
Just like with the lights, they have each their own type.

## Models
The model has three mandatory arguments:

 | Key        | Expected value     | What                                                        |
 |------------|--------------------|-------------------------------------------------------------|
 | `type`     | string (`"model"`) | The type of this object (should be `"model"`)               |
 | `material` | number             | The index of this model's material in the `materials` array |
 | `file`     | string             | The file path to the model                                  |

A note on the file path: it should be either an absolute path, or it should be relative to the directory where *cutrace is executed from*.
All example scenes expect that cutrace is executed from the root directory of this repository.

## Planes
The plane has four mandatory arguments:

 | Key        | Expected value     | What                                                        |
 |------------|--------------------|-------------------------------------------------------------|
 | `type`     | string (`"plane"`) | The type of this object (should be `"plane"`)               |
 | `material` | number             | The index of this plane's material in the `materials` array |
 | `point`    | vector             | The coordinates of any point on this plane                  |
 | `normal`   | vector             | The normal vector (direction) to this plane                 | 

## Triangles
A triangle has three mandatory arguments:

 | Key        | Expected value        | What                                                        |
 |------------|-----------------------|-------------------------------------------------------------|
 | `type`     | string (`"triangle"`) | The type of this object (should be `"triangle"`)            |
 | `material` | number                | The index of this model's material in the `materials` array |
 | `points`   | array                 | The coordinates of the three corners of this triangle       |

The points array should have exactly three values, and each should be a vector.

## Spheres
The sphere has four mandatory arguments:

 | Key        | Expected value      | What                                                        |
 |------------|---------------------|-------------------------------------------------------------|
 | `type`     | string (`"sphere"`) | The type of this object (should be `"sphere"`)              |
 | `material` | number              | The index of this plane's material in the `materials` array |
 | `center`   | vector              | The coordinates of the center point of this sphere          |
 | `radius`   | number              | The radius of this sphere                                   |

# Materials
Each material has five mandatory arguments:

 | Key            | Expected value | What                                              |
 |----------------|----------------|---------------------------------------------------|
 | `color`        | vector         | The material's base color                         |
 | `specular`     | number         | The material's specular (shininess) factor        |
 | `reflect`      | number         | The material's reflection (mirror-like) factor    |
 | `transparency` | number         | The material's transparency (translucency) factor |
 | `phong`        | number         | The material's Phong exponent                     |

Each of the values (including the color components) is expected to be in the `[0, 1]` inclusive interval. 
For the factors, the closer to 1, means the more shiny/mirror-like/translucent.

The only exception to the rule above is the Phong exponent.