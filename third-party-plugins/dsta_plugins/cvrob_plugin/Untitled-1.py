>>> import functools
>>> _original_init = MultiPartParser.__init__
>>> @functools.wraps(_original_init)
... def _patched_init(self, headers, stream, max_files=10_000, max_fields=10_000, **kwargs):
  _original_in...     _original_init(self, headers, stream, max_files=max_files, max_fields=max_fields, **kwargs)
... MultiPartParser.__init__ = _patched_init
  File "<stdin>", line 4
    MultiPartParser.__init__ = _patched_init
    ^^^^^^^^^^^^^^^
SyntaxError: invalid syntax
>>> def _patched_init(self, headers, stream, max_files=10_000, max_fields=10_000, **kwargs):
...     _original_init(self, headers, stream, max_files=max_files, max_fields=max_fields, **kwargs)
...
>>> MultiPartParser.__init__ = _patched_initMultiPartParser.__init__ = _patched_init
KeyboardInterrupt
>>> MultiPartParser.__init__ = _patched_init
>>> sig_after = inspect.signature(MultiPartParser.__init__)
sig_after)>>> print(sig_after)
(self, headers, stream, max_files=10000, max_fields=10000, **kwargs)
>>> exit()

kubectl port-forward svc/portal -n aiverify 3001:3000 & kubectl port-forward svc/apigw -n aiverify 4001:4000

from minio import Minio

client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

for file in files:
    client.fput_object(
        "datasets",
        f"all_images/{file.name}",
        file.path
    )

style={{color: '#111', backgroundColor: '#fff', border: '1px solid #ccc', padding: '8px 10px', borderRadius: '4px', }}











the file size for brittleness_carousel.html is still very large. there's 220 images in it, for reference. is that normal? 



 the "brittleness" that is defined here doesn't look like it means anything. "A score, B score, delta brittleness" won't mean anything to a layman. can you make it say something based on what the "brittleness" in this code means? 



for reference for how image classification was handled, for example in the matplotlib:



        axes[row, 0].imshow(imgA)
        axes[row, 0].axis("off")
        axes[row, 0].text(
            0.02, 0.98,
            f"image idx: {idx}\nA (before) | pred={classA}\n\npred_proba of {classA}={predA[1]:.3f}",
            transform=axes[row, 0].transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )
        
        # plt.tight_layout()

        axes[row, 1].imshow(imgB)
        axes[row, 1].axis("off")
        axes[row, 1].text(
            0.02, 0.98,
            f"image idx: {idx}\nB (after) | pred={classB}\npred_proba of {classB}={predB[1]:.3f}\npred_proba of {classA}={(predA[1]-res.brittleness):.3f} | Δ={res.brittleness:.3f}",
            transform=axes[row, 1].transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

