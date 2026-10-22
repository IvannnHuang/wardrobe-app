package dev.wardrobe.app.domain.cutout

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ClothesPiecesTest {
    // 1000-pixel "photos": labels + LAB colours per pixel.
    private val n = 1000
    private val black = floatArrayOf(12f, 0f, 0f)
    private val teal = floatArrayOf(48f, -28f, -8f)
    private val white = floatArrayOf(95f, 0f, 0f)
    private val brown = floatArrayOf(38f, 16f, 26f)

    private fun photo(vararg parts: Triple<IntRange, Int, FloatArray>): Pair<ByteArray, FloatArray> {
        val labels = ByteArray(n)
        val lab = FloatArray(n * 3)
        for ((range, label, colour) in parts) for (i in range) {
            labels[i] = label.toByte()
            for (c in 0..2) lab[i * 3 + c] = colour[c]
        }
        return labels to lab
    }

    /** Body figure: teal base top + base pants, bare feet. */
    private val base = photo(
        Triple(0 until 300, ClothesLabel.UPPER, teal),
        Triple(300 until 600, ClothesLabel.PANTS, teal),
    ).let { (labels, lab) -> learnBase(lab, labels) }

    @Test
    fun baseOutfitIsDroppedAndNewTopKept() {
        // black jacket open over the teal base top (teal shows in the opening), base pants, new brown shoes
        val (labels, lab) = photo(
            Triple(0 until 250, ClothesLabel.UPPER, black),
            Triple(250 until 300, ClothesLabel.UPPER, teal),
            Triple(300 until 600, ClothesLabel.PANTS, teal),
            Triple(600 until 620, ClothesLabel.LEFT_SHOE, brown),
        )
        val (pieces, notes) = extractPieces(lab, labels, base)
        assertEquals(listOf(PieceType.TOP, PieceType.SHOES), pieces.map { it.type })
        val top = pieces[0]
        assertEquals(250, top.keep.count { it })        // opening removed
        assertEquals(300, top.region.count { it })
        assertEquals(50f / 300, top.baseFrac, 1e-4f)
        assertTrue(notes.single().message.startsWith("bottom: base outfit"))
    }

    @Test
    fun accessoriesAlreadyWornOnTheBodyFigureAreDropped() {
        val withShoes = photo(
            Triple(0 until 300, ClothesLabel.UPPER, teal),
            Triple(600 until 620, ClothesLabel.RIGHT_SHOE, brown),
        ).let { (labels, lab) -> learnBase(lab, labels) }
        val (labels, lab) = photo(
            Triple(0 until 300, ClothesLabel.UPPER, white),
            Triple(600 until 620, ClothesLabel.LEFT_SHOE, brown),
        )
        val (pieces, notes) = extractPieces(lab, labels, withShoes)
        assertEquals(listOf(PieceType.TOP), pieces.map { it.type })
        assertEquals(listOf("shoes: already worn on body figure"), notes.map { it.message })
    }

    @Test
    fun lightingChangeIsCorrectedWithSkin() {
        val skin = floatArrayOf(65f, 15f, 18f)
        val bodyWithSkin = photo(
            Triple(0 until 300, ClothesLabel.UPPER, teal),
            Triple(300 until 600, ClothesLabel.PANTS, teal),
            Triple(600 until 1000, ClothesLabel.FACE, skin),
        ).let { (labels, lab) -> learnBase(lab, labels, PieceSettings(minSkinPixels = 100)) }
        // Same base pants, but this photo is much darker and bluer: everything shifted by (-20, -8, -12).
        fun dim(c: FloatArray) = floatArrayOf(c[0] - 20f, c[1] - 8f, c[2] - 12f)
        val (labels, lab) = photo(
            Triple(0 until 300, ClothesLabel.UPPER, dim(black)),
            Triple(300 until 600, ClothesLabel.PANTS, dim(teal)),
            Triple(600 until 1000, ClothesLabel.FACE, dim(skin)),
        )
        val settings = PieceSettings(minSkinPixels = 100)
        val (pieces, notes) = extractPieces(lab, labels, bodyWithSkin, settings)
        assertEquals(listOf(PieceType.TOP), pieces.map { it.type })
        assertTrue(notes.single().message.startsWith("bottom: base outfit"))
        // Without the skin reference the dimmed pants would no longer match the base colour.
        val noSkin = extractPieces(lab, labels, bodyWithSkin.copy(skin = null), settings).first
        assertTrue(noSkin.any { it.type == PieceType.BOTTOM })
    }

    @Test
    fun tinyPiecesAreIgnored() {
        val (labels, lab) = photo(Triple(0 until 1, ClothesLabel.HAT, black))
        assertTrue(extractPieces(lab, labels, base).first.isEmpty())
    }
}
