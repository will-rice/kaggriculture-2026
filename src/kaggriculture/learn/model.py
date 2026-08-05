"""The reference policy: Frog Parade's published shape, sized to our board.

Their Lux S3 solution is an 8-block 3x3 residual CNN at d_model 256, about 10M
parameters, with global features projected through an MLP and added to the
spatial tensor. Every verified winner across four competitions falls between 1M
and 20M parameters, and the two figures that suggested otherwise turned out to
be step counts.

Our board is 10x10 against their 24x24, so the same shape costs roughly a sixth
as much: the Phase 0 probe measured a 10M-parameter conv trunk at ~45ms of the
1000ms turn.

The readout follows them too: rather than pool the trunk and emit every unit's
op from one wide ``Linear``, each unit's logits are read from the trunk column
at the tile that unit is standing on, through one ``Linear`` shared by every
slot. Almost all of the parameters are in the trunk either way; what changes is
that the head can see position at all.
"""

import torch

from kaggriculture.learn.encoding import (
    SCALARS,
    TILE_PLANES,
    UNIT_OPS,
)

BLOCKS = 8
CHANNELS = 256


class Residual(torch.nn.Module):
    """One 3x3 residual block with squeeze-excitation and no normalisation.

    Toad Brigade's Lux S1 winner and Frog Parade's Lux S3 model both omit
    normalisation layers deliberately; this follows them rather than reasoning
    from first principles about a choice two winners already made.
    """

    def __init__(self, channels: int) -> None:
        """Build the block."""
        super().__init__()
        self.first = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.second = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.excite = torch.nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the block's output for a batch of feature maps."""
        residual = torch.nn.functional.relu(self.first(x))
        residual = self.second(residual)
        weights = torch.sigmoid(self.excite(residual.mean(dim=(2, 3))))
        return torch.nn.functional.relu(x + residual * weights[:, :, None, None])


class Policy(torch.nn.Module):
    """Board trunk plus a market branch, reading out one op per unit."""

    def __init__(self, blocks: int = BLOCKS, channels: int = CHANNELS) -> None:
        """Build the policy."""
        super().__init__()
        self.stem = torch.nn.Conv2d(TILE_PLANES, channels, 3, padding=1)
        self.market = torch.nn.Sequential(
            torch.nn.Linear(SCALARS, channels),
            torch.nn.ReLU(),
            torch.nn.Linear(channels, channels),
        )
        self.blocks = torch.nn.ModuleList(Residual(channels) for _ in range(blocks))
        self.head = torch.nn.Linear(channels, len(UNIT_OPS))

    def forward(
        self, board: torch.Tensor, scalars: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Return per-unit op logits.

        The market is projected and added to the spatial tensor rather than
        broadcast as constant planes: it is 28 numbers that decide the game and
        deserve their own capacity.

        Each unit reads the trunk at its own tile rather than at a global
        average. Pooling the trunk and emitting every slot from one ``Linear``
        gives slot *k* nothing but a board summary and its own index, so it can
        express what the day calls for but never what the unit is standing next
        to; measured, perturbing one board cell moved all ``MAX_UNITS`` slots.
        Gathering per position also makes the readout weights shared across
        slots, so which unit a slot holds stops mattering.

        Args:
            board: ``(batch, TILE_PLANES, BOARD, BOARD)`` planes.
            scalars: ``(batch, SCALARS)`` market and phase features.
            positions: ``(batch, MAX_UNITS)`` int64 flattened tile indices, in
                the order ``encode_units`` labels the units.

        Returns:
            ``(batch, MAX_UNITS, len(UNIT_OPS))`` logits.
        """
        features = self.stem(board) + self.market(scalars)[:, :, None, None]
        for block in self.blocks:
            features = block(features)
        columns = features.flatten(2)
        wanted = positions[:, None, :].tile(1, columns.shape[1], 1)
        return self.head(columns.gather(2, wanted).transpose(1, 2))
