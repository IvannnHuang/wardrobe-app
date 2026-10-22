package dev.wardrobe.app.domain.cutout

/** What to do with a piece after looking at all views of the garment. */
enum class VoteResult { KEEP, BASE_OUTFIT, TOO_FEW_VIEWS }

/**
 * The 4 views show the same garment, so they vote. A piece recognised as base outfit / already worn in at
 * least as many views as it was kept in is dropped everywhere (one bad photo, e.g. odd light, can't make
 * the base pants a "new garment"). A piece kept in only one of 3+ views is most likely a mistake.
 * [kept] / [dropped]: number of views per piece type.
 */
fun voteAcrossViews(kept: Map<PieceType, Int>, dropped: Map<PieceType, Int>, viewsProcessed: Int): Map<PieceType, VoteResult> =
    kept.mapValues { (type, k) ->
        val d = dropped[type] ?: 0
        when {
            d >= k -> VoteResult.BASE_OUTFIT
            k == 1 && viewsProcessed >= 3 -> VoteResult.TOO_FEW_VIEWS
            else -> VoteResult.KEEP
        }
    }
