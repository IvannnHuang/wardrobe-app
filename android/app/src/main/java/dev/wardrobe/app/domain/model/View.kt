package dev.wardrobe.app.domain.model

/** The 4 capture angles, in capture order. */
enum class View(val key: String) {
    FRONT("front"),
    RIGHT("right"),
    BACK("back"),
    LEFT("left"),
}
