# A Deep Learning post-processor with a perceptual loss function for video compression artefact removal
### Supplementary Data

Dataset available: [Link to follow]

---

The "models.py" file contains: Spatial Artefact Reduction Model, Temporal Artefact Reduction Model and ProxyVQA Model. 
These models can be instanciated by:

```
from models import SpatialSuppression, TemporalSuppression, VideoQualityAssessment
```

--- 

The savedModels.py folder shows how these models can be loaded with the weights presented in the literature. 
