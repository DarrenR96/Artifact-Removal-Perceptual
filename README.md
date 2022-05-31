# A Deep Learning post-processor with a perceptual loss function for video compression artefact removal
### Supplementary Data

Dataset available: [Link to follow](https://forms.office.com/Pages/ResponsePage.aspx?id=jb6V1Qaz9EWAZJ5bgvvlK088VywVtzhHrTQguQrCRblUNTEwSkc2TUY5TDJYRU1DWjBKSE5SSEFaUi4u)

---

The "models.py" file contains: Spatial Artefact Reduction Model, Temporal Artefact Reduction Model and ProxyVQA Model. 
These models can be instanciated by:

```
from models import SpatialSuppression, TemporalSuppression, VideoQualityAssessment
```

--- 

The savedModels.py folder shows how these models can be loaded with the weights presented in the literature. 
