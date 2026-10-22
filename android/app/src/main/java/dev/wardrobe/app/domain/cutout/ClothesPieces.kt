package dev.wardrobe.app.domain.cutout

import dev.wardrobe.app.domain.color.deltaE
import dev.wardrobe.app.domain.color.kMeans
import dev.wardrobe.app.domain.color.medianLab
import dev.wardrobe.app.domain.color.sampleLab

// Cut-out method C (see prototype/clothes.py, chosen by prototype/experiment.py): the clothes model labels
// every pixel; the base outfit is removed by colour, per region. Pure logic on flat arrays; the image
// work (model, cleanup, warping) is in pipeline/cutout.

/** Labels of the clothes model (SegFormer-B2 trained on ATR). */
object ClothesLabel {
    const val BACKGROUND = 0
    const val HAT = 1
    const val HAIR = 2
    const val UPPER = 4
    const val SKIRT = 5
    const val PANTS = 6
    const val DRESS = 7
    const val BELT = 8
    const val LEFT_SHOE = 9
    const val RIGHT_SHOE = 10
    const val FACE = 11
    const val LEFT_ARM = 14
    const val RIGHT_ARM = 15
    const val BAG = 16
    const val SCARF = 17
    const val COUNT = 18
}

enum class BaseRegion(vararg val labels: Int) {
    TOP(ClothesLabel.UPPER, ClothesLabel.DRESS),
    BOTTOM(ClothesLabel.SKIRT, ClothesLabel.PANTS),
}

/** Garment pieces the model can separate, and which base-outfit colour each is checked against. */
enum class PieceType(val key: String, val base: BaseRegion?, vararg val labels: Int) {
    TOP("top", BaseRegion.TOP, ClothesLabel.UPPER),
    DRESS("dress", BaseRegion.TOP, ClothesLabel.DRESS),
    BOTTOM("bottom", BaseRegion.BOTTOM, ClothesLabel.SKIRT, ClothesLabel.PANTS),
    SHOES("shoes", null, ClothesLabel.LEFT_SHOE, ClothesLabel.RIGHT_SHOE),
    HAT("hat", null, ClothesLabel.HAT),
    BAG("bag", null, ClothesLabel.BAG),
    BELT("belt", null, ClothesLabel.BELT),
    SCARF("scarf", null, ClothesLabel.SCARF),
}

/** Face + arms: the colour reference for lighting differences between photos. */
private val SKIN = intArrayOf(ClothesLabel.FACE, ClothesLabel.LEFT_ARM, ClothesLabel.RIGHT_ARM)

data class PieceSettings(
    /** LAB distance under which a pixel counts as base-outfit colour. */
    val baseDeltaE: Float = 16f,
    /** Lightness counts this much in colour distances (exposure changes mostly move lightness). */
    val lightnessWeight: Float = 0.5f,
    /** Minimum skin pixels needed to use skin as the lighting reference. */
    val minSkinPixels: Int = 500,
    /** A piece with at least this share of base colour is the base garment itself. */
    val basePieceFrac: Float = 0.6f,
    /** Pieces smaller than this share of the photo are ignored. */
    val minPieceFrac: Float = 0.002f,
)

/** What the base outfit looks like, learned from the body figure photo of the same view. */
data class BaseOutfit(
    val palettes: Map<BaseRegion, List<FloatArray>>,
    /** Accessories (shoes, hat, ...) already worn in the body figure, with their median colour. */
    val worn: Map<PieceType, FloatArray>,
    /** Median skin colour on the body figure (lighting reference), if enough skin is visible. */
    val skin: FloatArray? = null,
)

/** Colour distance with lightness down-weighted. */
fun weightedDeltaE(lab: FloatArray, i: Int, c: FloatArray, lightnessWeight: Float): Float {
    val dl = (lab[i * 3] - c[0]) * lightnessWeight
    val da = lab[i * 3 + 1] - c[1]
    val db = lab[i * 3 + 2] - c[2]
    return kotlin.math.sqrt(dl * dl + da * da + db * db)
}

fun IntArray.maskOf(labels: ByteArray): BooleanArray {
    val set = BooleanArray(ClothesLabel.COUNT).also { s -> forEach { s[it] = true } }
    return BooleanArray(labels.size) { set[labels[it].toInt() and 0xFF] }
}

fun learnBase(lab: FloatArray, labels: ByteArray, s: PieceSettings = PieceSettings()): BaseOutfit {
    val palettes = BaseRegion.entries.associateWith { region ->
        val mask = region.labels.maskOf(labels)
        if (mask.count { it } < 50) emptyList()
        else kMeans(sampleLab(lab, mask, 20_000), 3).filter { it.share >= 0.1f }.map { it.center }
    }
    val worn = PieceType.entries.filter { it.base == null }.mapNotNull { type ->
        val mask = type.labels.maskOf(labels)
        if (mask.count { it } < s.minPieceFrac * labels.size) null else medianLab(lab, mask)?.let { type to it }
    }.toMap()
    val skin = SKIN.maskOf(labels).takeIf { m -> m.count { it } >= s.minSkinPixels }?.let { medianLab(lab, it) }
    return BaseOutfit(palettes, worn, skin)
}

/** A garment piece before edge cleanup: pixels to keep, the model's region, and its base-colour share. */
class RawPiece(val type: PieceType, val keep: BooleanArray, val region: BooleanArray, val baseFrac: Float)

/** Splits a garment photo into pieces and removes the base outfit. Returns pieces + notes for the report. */
fun extractPieces(
    lab: FloatArray,
    labels: ByteArray,
    base: BaseOutfit,
    s: PieceSettings = PieceSettings(),
): Pair<List<RawPiece>, List<String>> {
    val n = labels.size
    val pieces = mutableListOf<RawPiece>()
    val notes = mutableListOf<String>()
    // Lighting: shift this photo's colours so my skin matches the body figure's skin (experiment_light.py).
    val skinMask = SKIN.maskOf(labels)
    val skin = if (base.skin != null && skinMask.count { it } >= s.minSkinPixels) medianLab(lab, skinMask) else null
    val lab = if (skin == null) lab else {
        val shift = FloatArray(3) { base.skin!![it] - skin[it] }
        FloatArray(lab.size) { lab[it] + shift[it % 3] }
    }
    val lw = s.lightnessWeight
    for (type in PieceType.entries) {
        val region = type.labels.maskOf(labels)
        val regionCount = region.count { it }
        if (regionCount < s.minPieceFrac * n) continue

        val worn = base.worn[type]
        val median = medianLab(lab, region)
        if (worn != null && median != null && weightedDeltaE(median, 0, worn, lw) < s.baseDeltaE) {
            notes += "${type.key}: already worn on body figure"
            continue
        }
        val palette = type.base?.let { base.palettes[it] }.orEmpty()
        if (palette.isEmpty()) {
            pieces += RawPiece(type, region, region, 0f)
            continue
        }
        var baseCount = 0
        val keep = BooleanArray(n) { i ->
            if (!region[i]) return@BooleanArray false
            val isBase = palette.any { weightedDeltaE(lab, i, it, lw) < s.baseDeltaE }
            if (isBase) baseCount++
            !isBase
        }
        val frac = baseCount.toFloat() / regionCount
        if (frac >= s.basePieceFrac) {
            notes += "${type.key}: base outfit (${(frac * 100).toInt()}% base colour)"
            continue
        }
        pieces += RawPiece(type, keep, region, frac)
    }
    return pieces to notes
}
