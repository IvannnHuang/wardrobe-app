package dev.wardrobe.app.domain.tags

import dev.wardrobe.app.domain.color.kMeans
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Synthetic figure (400 x 800 px) standing front-on, with garments drawn as unions of body capsules. */
class MeasuredTagsTest {
    private val w = 400
    private val h = 800

    private val lm = FloatArray(33 * 3).also { a ->
        fun set(i: Int, x: Float, y: Float) { a[i * 3] = x; a[i * 3 + 1] = y; a[i * 3 + 2] = 0.95f }
        set(0, 200f, 80f); set(7, 215f, 75f); set(8, 185f, 75f)
        set(11, 250f, 160f); set(12, 150f, 160f)      // shoulders (my left on the image right)
        set(13, 265f, 270f); set(14, 135f, 270f)      // elbows
        set(15, 270f, 370f); set(16, 130f, 370f)      // wrists
        for (i in intArrayOf(17, 19, 21)) set(i, 272f, 395f)
        for (i in intArrayOf(18, 20, 22)) set(i, 128f, 395f)
        set(23, 230f, 400f); set(24, 170f, 400f)      // hips
        set(25, 232f, 570f); set(26, 168f, 570f)      // knees
        set(27, 232f, 730f); set(28, 168f, 730f)      // ankles
        set(29, 230f, 750f); set(30, 170f, 750f); set(31, 245f, 760f); set(32, 155f, 760f)
    }

    private fun cap(a: Int, b: Int, r: Float) = Mask.capsule(w, h, lm[a * 3], lm[a * 3 + 1], lm[b * 3], lm[b * 3 + 1], r)
    private val torso = Mask.polygon(w, h, floatArrayOf(250f, 150f, 170f, 230f), floatArrayOf(160f, 160f, 400f, 400f))
    private val person = listOf(
        torso, Mask.circle(w, h, 200f, 80f, 45f),
        cap(11, 13, 25f), cap(13, 15, 22f), cap(12, 14, 25f), cap(14, 16, 22f),
        cap(23, 25, 35f), cap(25, 27, 28f), cap(24, 26, 35f), cap(26, 28, 28f),
        cap(29, 31, 15f), cap(30, 32, 15f),
    ).reduce { a, b -> a or b }

    private val regions = bodyRegions(lm, person)

    private fun longSleeveTop() = (torso or cap(11, 13, 30f) or cap(13, 15, 27f) or cap(12, 14, 30f) or cap(14, 16, 27f)) and
        Mask.polygon(w, h, floatArrayOf(0f, 400f, 400f, 0f), floatArrayOf(150f, 150f, 420f, 420f))

    private fun shortSleeveTop() = (torso or cap(11, 13, 30f) or cap(12, 14, 30f)) and
        Mask.polygon(w, h, floatArrayOf(0f, 400f, 400f, 0f), floatArrayOf(150f, 150f, 240f, 240f)) or torso

    private fun pants(bottomY: Float, legR: Float = 32f) = (cap(23, 25, legR) or cap(25, 27, legR) or cap(24, 26, legR) or cap(26, 28, legR) or
        Mask.polygon(w, h, floatArrayOf(165f, 235f, 235f, 165f), floatArrayOf(380f, 380f, 420f, 420f))) and
        Mask.polygon(w, h, floatArrayOf(0f, 400f, 400f, 0f), floatArrayOf(375f, 375f, bottomY, bottomY))

    @Test
    fun sleeveAndTopLength() {
        val long = longSleeveTop()
        val cov = coverage(long, regions)
        assertTrue("torso ${cov["torso"]}", cov.getValue("torso") > 0.9f)
        assertEquals("long", sleeveLength(cov).value)
        assertEquals("hip", topLength(long, lm).value)

        val short = shortSleeveTop()
        assertEquals("short", sleeveLength(coverage(short, regions)).value)
        assertEquals("sleeveless", sleeveLength(coverage(torso, regions)).value)
        val crop = torso and Mask.polygon(w, h, floatArrayOf(0f, 400f, 400f, 0f), floatArrayOf(150f, 150f, 300f, 300f))
        assertEquals("cropped", topLength(crop, lm).value)
    }

    @Test
    fun bottomsLengthAndCoverage() {
        assertEquals("full", bottomsLength(pants(760f), lm).value)
        assertEquals("knee", bottomsLength(pants(580f), lm).value)
        assertEquals("short", bottomsLength(pants(470f), lm).value)
        val cov = coverage(pants(760f), regions)
        assertTrue(cov.getValue("thighs") > 0.8f && cov.getValue("lowerLegs") > 0.9f)
        assertTrue(cov.getValue("torso") < 0.2f) // the waistband covers the bottom of the torso
    }

    @Test
    fun riseNeedsVisibleWaistband() {
        val p = pants(760f)
        assertEquals("unknown (waistband covered)", rise(p, lm, longSleeveTop()).value)
        val r = rise(p, lm, null)
        assertEquals("low", r.value) // top edge at 375 px, just above the hips (400)
    }

    @Test
    fun legShapeFromWidths() {
        val body = legWidths(person, lm)
        val t = torsoHeight(lm)
        assertEquals("skinny", legShape(legWidths(pants(760f, 29f), lm), t, body, t).value)
        assertEquals("wide", legShape(legWidths(pants(760f, 55f), lm), t, body, t).value)
        // same pants photographed from twice as far away: widths and torso both halve -> same answer
        val far = legWidths(pants(760f, 55f), lm).mapValues { it.value / 2 }
        assertEquals("wide", legShape(far, t / 2, body, t).value)
    }

    @Test
    fun sheernessCountsSmallHolesNotOpenings() {
        val region = longSleeveTop()
        val open = region.andNot(Mask.polygon(w, h, floatArrayOf(190f, 210f, 210f, 190f), floatArrayOf(160f, 160f, 400f, 400f)))
        assertEquals("opaque", sheerness(open, region).value)
        val dotted = Mask(w, h, BooleanArray(w * h) { i -> region.data[i] && !((i % w) % 6 == 0 && (i / w) % 6 == 0) })
        assertTrue(sheerness(dotted, region).measure!! > 0.02f)
    }

    @Test
    fun coloursAreNamedAndMerged() {
        val n = 1000
        val lab = FloatArray(n * 3) { i ->
            val p = i / 3
            when (i % 3) {
                0 -> if (p < 700) 10f + (p % 7) else 95f   // 70% black (two shades), 30% white
                else -> 0f
            }
        }
        val c = colours(lab, BooleanArray(n) { true })
        assertEquals(listOf("black", "white"), c.map { it.name })
        assertEquals(0.7f, c[0].share, 0.02f)
        assertEquals("neutral", c[0].temperature)
        assertEquals("red", colourName(floatArrayOf(50f, 66f, 46f)))
    }

    @Test
    fun kMeansSeparatesClearGroups() {
        val pts = FloatArray(600) { i -> if (i / 3 < 100) (if (i % 3 == 0) 20f else 0f) else (if (i % 3 == 0) 80f else 0f) }
        val clusters = kMeans(pts, 2)
        assertEquals(2, clusters.size)
        assertEquals(0.5f, clusters[0].share, 1e-6f)
        assertTrue(clusters.map { it.center[0] }.sorted().let { it[0] < 25 && it[1] > 75 })
    }
}
