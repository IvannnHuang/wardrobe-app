package dev.wardrobe.app.domain.cutout

import org.junit.Assert.assertEquals
import org.junit.Test

class ViewVoteTest {
    @Test
    fun viewsVote() {
        val result = voteAcrossViews(
            kept = mapOf(PieceType.TOP to 4, PieceType.BOTTOM to 1, PieceType.SHOES to 2, PieceType.HAT to 1),
            dropped = mapOf(PieceType.BOTTOM to 3, PieceType.SHOES to 2),
            viewsProcessed = 4,
        )
        assertEquals(VoteResult.KEEP, result[PieceType.TOP])
        assertEquals(VoteResult.BASE_OUTFIT, result[PieceType.BOTTOM])   // base in 3 views, kept in 1
        assertEquals(VoteResult.BASE_OUTFIT, result[PieceType.SHOES])    // tie: already worn wins
        assertEquals(VoteResult.TOO_FEW_VIEWS, result[PieceType.HAT])
    }

    @Test
    fun singleViewCapturesKeepTheirPieces() {
        assertEquals(VoteResult.KEEP, voteAcrossViews(mapOf(PieceType.TOP to 1), emptyMap(), 1)[PieceType.TOP])
    }
}
