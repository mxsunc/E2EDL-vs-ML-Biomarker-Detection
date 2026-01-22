import pandas as pd
import numpy as np

def align_dfB_to_dfA(dfA: pd.DataFrame, dfB: pd.DataFrame) -> pd.DataFrame:
    cols = list(dfA.columns)
    out = dfB.copy()

    for i, col in enumerate(cols):
        if col not in out.columns:
            left = out[cols[i-1]] if i > 0 and cols[i-1] in out.columns else None
            right = out[cols[i+1]] if i < len(cols)-1 and cols[i+1] in out.columns else None

            if left is not None and right is not None:
                # mean of neighbors (numeric-safe)
                l = pd.to_numeric(left, errors='coerce')
                r = pd.to_numeric(right, errors='coerce')
                newcol = (l + r) / 2
                # if non-numeric rows -> fall back to whichever neighbor is not NaN
                newcol = newcol.combine_first(left).combine_first(right)
            elif left is not None:
                newcol = left
            elif right is not None:
                newcol = right
            else:
                newcol = pd.Series(np.nan, index=out.index)

            out[col] = newcol

    # drop extras and enforce order of dfA
    out = out.reindex(columns=cols)
    return out