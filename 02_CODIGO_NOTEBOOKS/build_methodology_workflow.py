"""Draw the actual nested validation protocol; no generated imagery or model fits."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
if ROOT.name == '02_CODIGO_NOTEBOOKS':
    ROOT = ROOT.parent
OUT = ROOT / '03_RESULTADOS' if (ROOT / '01_DATASET').exists() else ROOT / 'project_artifacts'


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(122/25.4, 122/25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 122), ylim=(155, 0))
    ax.axis('off')
    navy, teal, grey = '#173B53', '#146B68', '#526270'

    def box(x, y, w, h, title, body, colour=navy, fill='#F2F6F9', body_size=7.3):
        ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0,rounding_size=1.5',
            facecolor=fill, edgecolor=colour, linewidth=.8))
        ax.text(x+w/2, y+3.6, title, ha='center', va='center', fontsize=8,
                fontweight='bold', color=colour)
        ax.text(x+w/2, y+(h+5)/2, body, ha='center', va='center', fontsize=body_size,
                color='#23313B', linespacing=1.32)

    def arrow(points, colour=grey, dashed=False):
        xs, ys = zip(*points)
        if len(points)>2:
            ax.plot(xs[:-1], ys[:-1], color=colour, lw=.85,
                    linestyle='--' if dashed else '-')
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
            mutation_scale=7, linewidth=.85, color=colour,
            linestyle='--' if dashed else '-'))

    box(4, 1, 114, 17, 'Public spectra and data audit',
        '13,452 spectra × 204 channels · 2,242 kernels · 90 dishes\n'
        'Six stages (0–5); identities used for grouping, not predictors')
    arrow([(61,18),(61,22)])
    box(4, 22, 114, 12, 'Five outer dish-disjoint folds',
        'All kernels and stages from each dish stay together', body_size=7.5)
    arrow([(31,34),(31,39)],teal)
    arrow([(95,34),(95,39)],grey)
    ax.add_patch(FancyBboxPatch((2,37),67,70,boxstyle='round,pad=0,rounding_size=2',
        facecolor='#F4FAF8',edgecolor=teal,linestyle='--',linewidth=.8))
    box(5,39,61,11,'Outer training','72 dishes per fold',teal,'#E6F2ED',7.5)
    box(77,39,41,17,'Held-out test','18 dishes per fold\nNo fitting or tuning',grey,'#F3F4F5')
    arrow([(35.5,50),(35.5,54)],teal)
    box(5,54,61,19,'Four inner dish-grouped folds',
        'PLSR / SVR: raw, SNV or first derivative\nChannel scaling fitted on training only\nExtra Trees: raw spectra, no scaler',teal,'#FFFFFF',7)
    arrow([(35.5,73),(35.5,77)],teal)
    box(5,77,61,12,'Select within each model family',
        'PLSR / RBF SVR / Extra Trees · inner MAE',teal,'#FFFFFF',7)
    arrow([(35.5,89),(35.5,93)],teal)
    box(5,93,61,12,'Refit selected pipeline','All 72 outer-training dishes',teal,'#E6F2ED',7.5)
    box(77,77,41,22,'PLS interpretation',
        'Training-only VIP\nand signed coefficients\nNo outer-test spectra',teal,'#F1F8F5',7.1)
    arrow([(66,99),(72,99),(72,88),(77,88)],teal,True)
    arrow([(97.5,56),(120,56),(120,110),(98,110),(98,114)],grey)
    arrow([(35.5,105),(35.5,114)],teal)
    box(4,114,114,12,'Predict held-out spectra; concatenate five outer folds',
        'One out-of-fold prediction per observation and model family',body_size=7.2)
    arrow([(31,126),(31,130)])
    arrow([(92,126),(92,130)])
    box(4,130,55,14,'Primary evaluation',
        'All stages + post-moisture subset\nPaired dish bootstrap; fixed OOF errors',body_size=6.8)
    box(63,130,55,14,'Exploratory diagnostics',
        'Separate partition / exclusion / C runs\nExploratory; no primary-model reselection',body_size=6.8)
    ax.text(61,149,'Saved predictions, search tables and profiles → executable audit package',
        ha='center',va='center',fontsize=7.2,fontweight='bold',color=navy)
    ax.text(61,153,'Scope: one acquisition sequence; stage–session confounding remains unresolved',
        ha='center',va='center',fontsize=6.8,color=grey)
    # Atomic replacement protects complete assets during workspace synchronization.
    folder = OUT / 'revision_r1'
    folder.mkdir(parents=True,exist_ok=True)
    for extension in ['svg','png']:
        target = folder / ('fig_methodology_workflow.'+extension)
        tmp = target.with_name(target.name+'.tmp')
        fig.savefig(tmp,format=extension,dpi=600,facecolor='white')
        tmp.replace(target)
        print(target)
    plt.close(fig)


if __name__ == '__main__':
    main()
