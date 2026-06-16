export const ALL_AUGMENTATIONS = [
  { category: "Blur",                    name: "Gaussian Blur",          keys: ["GaussianBlur"],                   description: ["Applies a Gaussian kernel to smooth the image and reduce high-frequency details."], realLife: "Out-of-focus camera lens, shallow depth of field, slight camera shake." },
  { category: "Blur",                    name: "Motion Blur",            keys: [],                                 description: ["Blurs the image along a direction to simulate movement."], realLife: "Camera movement during exposure, fast-moving objects." },
  { category: "Blur",                    name: "Defocus Blur",           keys: [],                                 description: ["Simulates optical defocus from incorrect focal length."], realLife: "Autofocus failure, lens miscalibration." },
  { category: "Digital",                 name: "Brightness",             keys: ["BrightnessUp", "BrightnessDown"], description: ["Uniformly increases or decreases pixel intensity."], realLife: "Overexposure or underexposure due to lighting conditions." },
  { category: "Digital",                 name: "Contrast",               keys: ["Contrast"],                       description: ["Adjusts the difference between dark and light regions."], realLife: "Poor lighting, camera auto-adjustment errors." },
  { category: "Digital",                 name: "Perspective Transform",  keys: ["Perspective"],                    description: ["Warps the image to simulate viewpoint change."], realLife: "Object viewed from different camera angles or positions." },
  { category: "Digital",                 name: "Image Compression",      keys: ["Compression"],                    description: ["Introduces compression artifacts (blocking, ringing)."], realLife: "JPEG compression during storage, messaging apps, streaming." },
  { category: "Digital",                 name: "Erasing (Occlusion)",    keys: ["Erasing"],                        description: ["Randomly removes or masks regions of the image."], realLife: "Objects partially blocking the camera (e.g., pedestrians, poles, dirt)." },
  { category: "Digital",                 name: "Saturation",             keys: [],                                 description: ["Changes the intensity of colours."], realLife: "Different lighting environments, white balance shifts." },
  { category: "Weather",                 name: "Rain",                   keys: ["Rain"],                           description: ["Adds streak-like artifacts simulating rainfall."], realLife: "Outdoor surveillance during rain." },
  { category: "Weather",                 name: "Snow",                   keys: [],                                 description: ["Adds white particle noise simulating snowfall."], realLife: "Outdoor cameras during snowstorms." },
  { category: "Weather",                 name: "Fog / Haze",             keys: [],                                 description: ["Reduces contrast and adds atmospheric blur."], realLife: "Humid weather, pollution, mist." },
  { category: "Noise",                   name: "Gaussian Noise",         keys: ["GaussianNoise"],                  description: ["Adds random pixel-wise noise sampled from a Gaussian distribution."], realLife: "Low-light sensor noise, high ISO settings." },
  { category: "Noise",                   name: "Salt-and-Pepper Noise",  keys: [],                                 description: ["Random black and white pixels appear."], realLife: "Transmission errors, faulty sensors." },
  { category: "Noise",                   name: "Speckle Noise",          keys: [],                                 description: ["Multiplicative noise affecting pixel intensities."], realLife: "Radar imaging, medical ultrasound imaging." },
  { category: "Directional / Geometric", name: "Shear",                  keys: ["Shear"],                          description: ["Slants the image along one axis."], realLife: "Camera misalignment or rolling shutter effects." },
  { category: "Directional / Geometric", name: "Translate",              keys: ["Translate"],                      description: ["Shifts the image along the x or y axis."], realLife: "Object shifts within the frame, tracking misalignment." },
  { category: "Directional / Geometric", name: "Scale",                  keys: ["ScaleUp", "ScaleDown"],           description: ["Zooms in or out of the image."], realLife: "Object distance changes from the camera." },
  { category: "Directional / Geometric", name: "Rotate",                 keys: ["Rotate"],                         description: ["Rotates the image by a given angle."], realLife: "Camera tilt, handheld shooting." },  
  { category: "Directional / Geometric", name: "Crop",                   keys: [],                                 description: ["Removes outer regions of the image."], realLife: "Object partially outside the field of view." },
];

export const ALL_AUGMENTATIONS_LONG = [
  {
    name: "Gaussian Blur",
    keys: ["GaussianBlur"],
    longDescription: [`Gaussian blur manifests as a soft, uniform smearing of detail across the image — visually similar to squinting your eyes or looking through frosted glass. In real-world deployment, this corruption is most representative of:

• Shallow depth of field — when a subject sits outside the camera's focal plane, it renders with this characteristic smooth blur. Common in close-range inspection systems or cameras with wide apertures.
• Slight camera shake — minor vibrations during a long exposure produce a diffuse, non-directional blur that closely approximates a Gaussian kernel.
• Low-resolution upscaling — images enlarged from a lower native resolution often exhibit Gaussian-like softness due to interpolation.
• Lens imperfections — cheap or worn lenses scatter light more broadly, reducing the sharpness of fine details in predictable ways.
• Post-processing pipelines — many imaging systems apply a mild Gaussian smooth to reduce noise before storage, which degrades edge sharpness as a side effect.

What poor performance here tells you: Your model depends heavily on sharp edges and fine textures to make decisions. It may struggle with any camera system that isn't well-focused, or in pipelines where pre-processing smooths the image before inference.`],
  },
  {
    name: "Motion Blur",
    keys: [],
    longDescription: [`Motion blur appears as a directional smear — pixels trail off in one direction as though the image was dragged across the sensor during capture. It is one of the most common real-world degradations for cameras imaging dynamic scenes. This corruption is most representative of:

• Fast-moving subjects — vehicles, athletes, machinery, or animals moving quickly relative to the camera's shutter speed produce characteristic streak artifacts across the object.
• Camera panning — surveillance or handheld cameras that follow a moving subject blur the background while the subject stays sharp (or vice versa if tracking is imperfect).
• Long exposure in low light — cameras compensate for darkness by keeping the shutter open longer, making any movement — subject or camera — result in blur.
• High-speed industrial vision — conveyor belt inspection systems or traffic cameras often capture objects mid-motion at insufficient shutter speeds.
• Drones and mobile platforms — vibrations and flight movement introduce subtle directional blur even at normal shutter speeds.

What poor performance here tells you: Your model likely struggles in any deployment involving fast-moving objects or a moving camera platform. This is critical for autonomous vehicles, sports analytics, and any outdoor surveillance with traffic.`],
  },
  {
    name: "Defocus Blur",
    keys: [],
    longDescription: [`Defocus blur produces a circular, disc-like smearing of the image — rounder and more uniform than motion blur, resembling a "bokeh" effect on out-of-focus regions. It is caused by optical rather than motion-based factors. This corruption is most representative of:

• Autofocus failure — cameras that fail to lock focus correctly, particularly in low-contrast scenes or when the subject moves rapidly, produce this kind of blur.
• Fixed-focus cameras at unexpected distances — embedded cameras (e.g. on ATMs, doorbells, or industrial rigs) are calibrated for a specific focal distance; objects too close or too far render defocused.
• Lens miscalibration — over time, zoom lenses or cameras with physical adjustment rings can drift out of calibration, softening the entire image plane.
• Multi-camera rigs — stereo or multi-view setups with mismatched focal lengths produce defocus artifacts on one or more views.
• Depth-of-field variation — a model trained on images where the full scene is sharp may encounter real data where backgrounds (or foregrounds) are intentionally defocused.

What poor performance here tells you: Your model is brittle to optical imperfections and may degrade in any deployment where the camera hardware is not regularly maintained or precisely calibrated.`],
  },
  {
    name: "Brightness",
    keys: ["BrightnessUp", "BrightnessDown"],
    longDescription: [`Brightness shifts uniformly lighten or darken the entire image — think of it as turning up or down the exposure dial on a camera. It is among the most pervasive real-world variations a deployed model will encounter. This corruption is most representative of:

• Time-of-day variation — a camera fixed in place will image the same scene very differently at dawn, noon, and dusk. Shadows deepen, highlights blow out, and overall intensity changes dramatically.
• Overexposure — bright scenes or incorrectly set exposure parameters wash out detail in highlights, compressing information into a narrow bright range.
• Underexposure — indoor cameras, tunnels, or night-time scenarios produce dark images where detail is compressed into the low end of the pixel range.
• Automatic exposure drift — cameras with auto-exposure can oscillate between frames, especially when transitioning between light and shadow zones (e.g. a car entering a tunnel).
• Regional infrastructure variation — models deployed across geographies encounter different ambient light levels based on latitude, season, and local weather.

What poor performance here tells you: Your model has likely overfit to a narrow brightness regime in training data. It may fail in any deployment that spans multiple times of day, seasons, or indoor/outdoor transitions.`],
  },
  {
    name: "Contrast",
    keys: ["Contrast"],
    longDescription: [`Contrast reduction or boosting compresses or stretches the range of pixel values — low contrast makes images look flat and washed-out, while high contrast causes harsh clipping of detail in shadows and highlights. This corruption is most representative of:

• Overcast or flat lighting — diffuse cloud cover eliminates shadows and reduces scene contrast, making objects harder to distinguish from their backgrounds.
• Camera auto-contrast errors — AGC (automatic gain control) circuits can miscalibrate, particularly during scene transitions, producing frames with abnormal tonal ranges.
• Lens flare and glare — bright light sources entering the lens reduce local contrast across the image, particularly near the light source.
• Foggy or hazy conditions — atmospheric scattering reduces the apparent contrast of distant objects (related to fog/haze, but contrast alone captures the tonal flattening).
• Display-calibrated cameras — some cameras apply proprietary tone curves that boost apparent contrast for human viewing but distort the distribution in ways that confuse models.

What poor performance here tells you: Your model struggles when the visual separation between objects and their backgrounds is reduced. This has direct implications for any scene with challenging lighting, flat surfaces, or low-saturation environments.`],
  },
  {
    name: "Perspective Transform",
    keys: ["Perspective"],
    longDescription: [`A perspective transform warps the image as though it were being viewed from a different angle — objects tilt, recede, or appear skewed relative to their original orientation. In deployment, perspective shifts are almost inevitable when camera geometry isn't perfectly controlled. This corruption is most representative of:

• Non-standard camera mounting angles — in surveillance or retail analytics, cameras are rarely mounted at the exact angle assumed during model training. Even a few degrees of tilt changes object aspect ratios and orientations meaningfully.
• Mobile and handheld capture — users photographing documents, products, or scenes from non-frontal angles introduce perspective distortion into otherwise normal images.
• Aerial and drone imagery — cameras mounted on UAVs looking down at slight angles produce perspective transforms relative to the nadir view typical in satellite-style datasets.
• Wide-angle lens distortion — extreme wide-angle lenses introduce a form of perspective warp particularly at image edges, affecting object shape.
• Cross-camera generalisation — a model trained on images from one camera angle often degrades when deployed on a rig with a different physical geometry.

What poor performance here tells you: Your model has not generalised to viewpoint variation. It may be learning pose-specific features rather than view-invariant representations, which is a significant risk in any deployment where camera placement is not rigidly controlled.`],
  },
  {
    name: "Image Compression",
    keys: ["Compression"],
    longDescription: [`Compression artifacts appear as blocky grid patterns, ringing halos around edges, and smearing of fine detail — the distinctive signature of aggressive JPEG or video codec compression. This is one of the most ubiquitous degradations in real-world image pipelines. This corruption is most representative of:

• Social media and messaging pipelines — images shared via WhatsApp, Telegram, Instagram, or WeChat are re-compressed, often aggressively, before being re-displayed or further processed.
• Video frame extraction — frames extracted from compressed video (H.264, H.265) carry codec artifacts, especially in low-bitrate streams from IP cameras or dashcams.
• Edge device storage — embedded systems with limited storage capacity (dashcams, IoT cameras, body cameras) often apply heavy JPEG compression before saving to flash.
• Network-transmitted images — images sent over low-bandwidth connections are often compressed at source to reduce transmission time, particularly in remote monitoring applications.
• Database archival — historical image archives frequently used for training or evaluation were often stored with lossy compression years before the current model was designed.

What poor performance here tells you: Your model is likely seeing cleaner images during training than it will in production. If your inference pipeline involves any re-encoding or transmission step, compression robustness is not optional.`],
  },
  {
    name: "Erasing (Occlusion)",
    keys: ["Erasing"],
    longDescription: [`Erasing simulates occlusion by randomly masking out rectangular or irregular regions of the image — replacing them with noise, a mean value, or black patches. Visually, parts of the object or scene simply disappear. This corruption directly models one of the most common failure modes in real-world computer vision. This corruption is most representative of:

• Partial object occlusion — in crowded scenes, objects of interest (people, vehicles, products) are routinely blocked by other objects: poles, other pedestrians, parked cars, or shopping carts.
• Camera obstructions — dirt, rain droplets, insects, or physical damage on the camera lens create persistent masked regions in every frame.
• Foreground interference — in agricultural, warehouse, or outdoor settings, vegetation, machinery, or shelving frequently occlude the objects being monitored.
• Edge-of-frame cropping — objects moving through the camera's field of view enter and exit partially, with significant portions cut off at the image boundary.
• Privacy masking — some camera systems apply fixed privacy zones (e.g. masking windows in a store) that blank out portions of the scene permanently.

What poor performance here tells you: Your model may be relying on specific regions — faces, logos, license plates — that are often the first to be occluded in real deployments. Robustness to erasing indicates the model uses distributed, holistic features rather than a single dominant region.`],
  },
  {
    name: "Saturation",
    keys: [],
    longDescription: [`Saturation changes the intensity of colours in the image — high saturation makes colours vivid and oversaturated, while low saturation pushes the image toward grayscale. Because colour information is frequently used as a discriminative cue, saturation shifts can significantly affect model behaviour. This corruption is most representative of:

• White balance errors — cameras that miscalibrate white balance under artificial lighting (sodium vapour, fluorescent, LED) shift the colour cast of the entire scene, effectively altering perceived saturation.
• Different lighting environments — outdoor light at golden hour, overcast noon, and under shade produce very different colour saturation profiles for the same scene.
• Camera-to-camera variation — even identical camera models from the same manufacturer apply slightly different ISP (image signal processor) colour tuning, resulting in saturation differences across a fleet.
• Seasonal and environmental variation — greenery in summer versus winter, wet versus dry surfaces, and dusty versus clean environments all shift the colour saturation of a scene.
• Display and rendering pipelines — images pre-processed with colour enhancement filters (common in consumer devices) may arrive at inference with boosted saturation relative to training data.

What poor performance here tells you: Your model may be using colour as a shortcut rather than learning shape or texture features. Saturation sensitivity is a strong indicator of colour bias, which is particularly problematic when deploying across different geographies, lighting conditions, or camera hardware.`],
  },
  {
    name: "Rain",
    keys: ["Rain"],
    longDescription: [`Rain corruption adds semi-transparent streak-like artifacts across the image, simulating the visual effect of rainfall in front of the camera lens. The streaks vary in angle, density, and thickness depending on wind speed and rainfall intensity. This corruption is most representative of:

• Outdoor surveillance cameras — fixed cameras monitoring parking lots, building entrances, roads, or public spaces are exposed to rainfall year-round without shelter.
• Autonomous vehicle perception — front-facing cameras on cars and trucks encounter rain frequently; dense rainfall can degrade detection of pedestrians, traffic signs, and lane markings.
• Traffic monitoring systems — roadside cameras used for vehicle counting and incident detection must operate through all weather conditions, including heavy rain.
• Agricultural and environmental monitoring — field cameras monitoring crops, water levels, or wildlife are exposed to rain with no human intervention possible.
• Sports and event cameras — outdoor broadcast or tracking cameras at stadiums and arenas must continue operating through rain without opportunity to pause collection.

What poor performance here tells you: Your model is likely not robust to real-world outdoor deployment. Poor rain performance is especially critical for safety-related applications like autonomous driving or perimeter security, where degraded weather cannot be treated as a reason to halt inference.`],
  },
  {
    name: "Snow",
    keys: [],
    longDescription: [`Snow corruption overlays small white particles and a general brightness increase across the image, simulating falling snow or snow accumulation on the lens. Unlike rain, snowfall can also partially coat the camera housing, introducing persistent white regions. This corruption is most representative of:

• Winter outdoor surveillance — cameras in high-latitude or high-altitude environments routinely operate through snowfall, sometimes for months at a time.
• Autonomous and assisted driving in snowy climates — snow obscures lane markings, traffic signs, and road edges, creating some of the most challenging perception conditions for vehicle systems.
• Ski area and mountain monitoring — avalanche detection, ski patrol, and resort security cameras operate in environments where heavy snowfall is routine.
• Utility and infrastructure inspection — power lines, bridges, and pipelines in cold climates are monitored by cameras that must remain operational during snowstorms.
• Border and perimeter security — security cameras in northern climates cannot be taken offline during winter weather.

What poor performance here tells you: Your model likely lacks exposure to winter weather data in training. For any deployment above certain latitudes or altitudes, snow robustness is not edge-case hardening — it is a fundamental operational requirement.`],
  },
  {
    name: "Fog / Haze",
    keys: [],
    longDescription: [`Fog and haze reduce contrast uniformly across the image, adding a milky atmospheric layer that is thicker with distance. Unlike blur, the image retains some sharpness of nearby objects while distant detail is progressively obscured. This corruption is most representative of:

• Early morning conditions — ground fog is most common at dawn and is a regular occurrence for 24/7 outdoor cameras, particularly near water, valleys, or open fields.
• Air pollution and smog — urban cameras in cities with high particulate matter or industrial emissions regularly capture hazy images that degrade visibility.
• Coastal and maritime environments — cameras operating near oceans, ports, or estuaries frequently encounter sea mist and salt haze.
• Forest fires and smoke — wildfire smoke creates dense, uneven haze that can render outdoor cameras nearly unusable at high concentrations.
• Humid tropical climates — in Southeast Asia, Central America, and similar regions, atmospheric humidity creates persistent haze that affects all outdoor imagery.

What poor performance here tells you: Your model heavily relies on long-range features and fine contrast differences to make classifications. In hazy conditions, only nearby, high-contrast information survives — a model that performs well through fog has learned features that are genuinely close-range and contrast-independent.`],
  },
  {
    name: "Gaussian Noise",
    keys: ["GaussianNoise"],
    longDescription: [`Gaussian noise manifests as random, independent variations in pixel intensity across the image — visually similar to the "grain" or "static" you'd see on an old television. In real-world deployment, this corruption is most representative of:

• Low-light or night-time imaging — cameras compensate for low light by boosting sensor gain (ISO), which introduces electronic noise with roughly Gaussian characteristics. A model deployed in 24/7 surveillance or autonomous driving must handle this.
• Low-cost or embedded sensors — cheaper camera hardware has worse signal-to-noise ratios. If your model runs on edge devices (e.g. IoT cameras, mobile phones, drones), sensor noise is a realistic concern.
• Long-distance transmission — image data transmitted over noisy communication channels (e.g. satellite imagery, medical imaging over networks) can accumulate noise artifacts.
• High-speed imaging — short exposure times reduce the amount of light captured per frame, increasing shot noise, which approximates Gaussian noise at high intensities.

What poor performance here tells you: Your model may struggle in low-light deployments or with budget hardware, even if it performs well in controlled, well-lit conditions.`],
  },
  {
    name: "Salt-and-Pepper Noise",
    keys: [],
    longDescription: [`Salt-and-pepper noise appears as randomly scattered pure-white and pure-black pixels sprinkled across the image — as though someone had thrown grains of salt and pepper onto the picture. Unlike Gaussian noise, individual affected pixels are extreme outliers rather than subtle variations. This corruption is most representative of:

• Faulty or ageing sensors — individual sensor elements (pixels) in a camera array can fail permanently, producing stuck-at-max or stuck-at-zero outputs that manifest as persistent bright or dark spots.
• Transmission bit errors — digital image data transmitted over unreliable channels (e.g. radio links, corrupted network packets) can have individual bits flipped, producing isolated extreme pixel values.
• Cosmic ray interference — in high-altitude or space environments, ionising radiation can flip individual pixel values in sensor arrays, producing scattered bright spikes.
• Memory corruption — errors in image buffer memory or flash storage can corrupt isolated pixel values before they reach the inference pipeline.
• Older or damaged camera hardware — cameras that have exceeded their operational lifespan or been physically damaged frequently develop increasing numbers of dead or hot pixels.

What poor performance here tells you: Your model is sensitive to pixel-level outliers. If any of your deployment hardware is ageing, operating in harsh environments, or transmitting over unreliable channels, salt-and-pepper robustness is a practical operational concern rather than a synthetic stress test.`],
  },
  {
    name: "Speckle Noise",
    keys: [],
    longDescription: [`Speckle noise is a granular, multiplicative noise that modulates pixel intensities rather than adding uniform random offsets — brighter regions exhibit more noise than darker ones, creating a mottled or blotchy texture across the image. This corruption is most representative of:

• Radar and SAR imagery — synthetic aperture radar images inherently exhibit speckle as a result of coherent interference during image formation. Any model processing radar-derived imagery must handle this.
• Medical ultrasound — speckle is a fundamental characteristic of ultrasound imaging, arising from coherent scattering in biological tissue. Models deployed in clinical or remote diagnostic settings encounter this regularly.
• Laser-based imaging systems — LiDAR-derived intensity images and structured light systems exhibit speckle-like patterns due to coherent illumination.
• Thermal and infrared cameras — certain IR sensor types introduce multiplicative noise that resembles speckle in its statistical structure.
• Multispectral and hyperspectral sensors — sensors operating outside the visible spectrum often have worse signal-to-noise ratios, and noise in these bands can exhibit multiplicative characteristics.

What poor performance here tells you: Your model may be limited to clean optical imagery and is not ready for deployment in medical imaging, remote sensing, or non-visible-spectrum applications where speckle noise is not an artifact to be corrected but an inherent property of the data.`],
  },
  {
    name: "Shear",
    keys: ["Shear"],
    longDescription: [`Shear distortion slants the image along one axis — objects that should appear upright lean to one side, as though the image were printed on rubber and pulled from one corner. This geometric corruption is subtler than rotation but can significantly alter the apparent shape of objects. This corruption is most representative of:

• Rolling shutter distortion — cameras using CMOS sensors with rolling shutters read pixel rows sequentially rather than simultaneously. Fast lateral movement of the camera or subject causes a characteristic shear artifact in the image.
• Camera mounting misalignment — cameras not perfectly perpendicular to their intended axis of view produce a mild but consistent shear across the image that compounds with distance from the optical centre.
• Wide-angle lens distortion — certain lens geometries, particularly anamorphic or cylindrical lenses, introduce shear-like distortions at image edges.
• Document and receipt scanning — handheld or flatbed scans of paper documents frequently introduce shear if the document is not placed perfectly parallel to the scan axis.
• Satellite and aerial imagery — orbital mechanics and sensor geometry in push-broom satellite sensors can introduce shear artifacts that must be corrected, but often persist in raw image archives.

What poor performance here tells you: Your model likely relies on the upright, axis-aligned appearance of objects during training and has not learned to handle the geometric irregularities introduced by real camera hardware and motion dynamics.`],
  },
  {
    name: "Translate",
    keys: ["Translate"],
    longDescription: [`Translation shifts the entire image — or the objects within it — along the horizontal or vertical axis, placing subjects off-centre or partially out of frame. While simple in concept, translation sensitivity reveals fundamental assumptions baked into how a model processes spatial position. This corruption is most representative of:

• Tracking system jitter — in active tracking applications (PTZ cameras, drone gimbals), the subject is rarely perfectly centred in every frame. Slight latency in the tracking loop introduces consistent translation offsets.
• Object position variation — in fixed-camera deployments, objects of interest do not always appear in the same position. A pedestrian may cross anywhere in a corridor; a vehicle may park anywhere in a lot.
• Image stabilisation drift — even cameras with electronic image stabilisation (EIS) produce subtle frame-to-frame translation as the stabilisation algorithm compensates for motion.
• Sensor alignment variation — in multi-camera rigs or stereo systems, slight misregistration between cameras introduces translation offsets between corresponding views.
• Data ingestion pipelines — images from different sources may have different implicit origins or coordinate conventions, effectively translating content relative to what the model expects.

What poor performance here tells you: Your model may be positionally biased — expecting the subject to appear in a specific region of the image, likely the centre or a position typical in training data. This is a common failure mode in models trained without sufficient spatial augmentation.`],
  },
  {
    name: "Scale",
    keys: ["ScaleUp", "ScaleDown"],
    longDescription: [`Scale augmentation zooms into or out of the image, changing the apparent size of objects relative to the frame. Scaling up makes objects larger and crops peripheral context; scaling down shrinks objects and introduces more background. Object size variation is one of the most fundamental challenges in real-world deployment. This corruption is most representative of:

• Variable subject distance — the same camera will image a person at 1 metre very differently than at 10 metres. Any deployment where subjects can approach or recede from the camera must handle this.
• Zoom lens variation — cameras with motorised zoom can change focal length between captures, altering object scale without any movement of the subject.
• Multi-site deployment — deploying the same model across cameras with different focal lengths (and therefore different fields of view) means the same physical object appears at different scales per-camera.
• Aerial altitude variation — drone-based models encounter objects at vastly different scales depending on flight altitude, which changes with mission profile or wind conditions.
• Resolution normalisation — when images from cameras with different sensor resolutions are resized to a common input dimension, objects implicitly change scale relative to the training distribution.

What poor performance here tells you: Your model lacks scale invariance and may perform well only within a narrow size range corresponding to training data. This is particularly limiting for any deployment where camera placement or subject proximity varies.`],
  },
  {
    name: "Rotate",
    keys: ["Rotate"],
    longDescription: [`Rotation tilts the image by a given angle — objects appear at an angle to the vertical, with the degree of tilt ranging from subtle (a few degrees of camera lean) to extreme (90° or 180° flips). Most real-world cameras are not perfectly level, and many deployment contexts involve significant rotation. This corruption is most representative of:

• Handheld capture — smartphones and handheld cameras are rarely held perfectly level. Even with software correction, small residual rotation angles persist in captured images.
• Camera mounting imprecision — cameras physically mounted to walls, poles, or vehicles are not always perfectly aligned to a true horizontal plane. Even small installation errors accumulate across large camera fleets.
• Vehicle and drone-mounted cameras — cameras on tilting platforms (dashcams on uneven roads, cameras on maneuvering drones) encounter continuous rotation relative to the scene.
• Medical imaging — X-ray, pathology slide scanning, and endoscope images may be captured at arbitrary orientations without a canonical "up" direction.
• Satellite imagery — images from satellites in different orbital inclinations can arrive rotated relative to north-aligned map conventions, requiring models to handle arbitrary orientation.

What poor performance here tells you: Your model has likely learned orientation-specific features — upright people, horizontal roads, vertically-oriented text — that fail when the camera is not level. This is a significant risk in any deployment without rigid, regularly inspected camera mounting.`],
  },
  {
    name: "Crop",
    keys: [],
    longDescription: [`Cropping removes the outer edges of the image, retaining only a central or offset sub-region. While the remaining content is clean and undistorted, objects near the original image boundary may be cut off entirely or appear only partially. This tests whether a model can reason about incomplete scenes. This corruption is most representative of:

• Object at field-of-view boundary — in any deployment with a fixed camera, subjects of interest frequently enter or exit the frame, with only a portion visible at any given moment.
• Resolution downsampling pipelines — images preprocessed for transmission or storage are sometimes cropped to a standard aspect ratio before being resized, removing content at the edges.
• Pan-tilt-zoom errors — PTZ cameras that move to track a subject can overshoot, resulting in the subject being partially or fully outside the captured frame.
• Multi-crop ensemble inference — some inference pipelines break an image into overlapping crops for efficiency, meaning individual crops may contain only part of any given object.
• Inconsistent field-of-view across cameras — different camera models with different sensor sizes have different fields of view, effectively cropping the scene differently even when physically co-located.

What poor performance here tells you: Your model likely depends on seeing complete objects and loses confidence rapidly when parts are missing. This matters in any deployment where objects are frequently near the image boundary — which, in practice, is nearly every real fixed-camera setup.`],
  },
];
