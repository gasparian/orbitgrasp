from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="DetectResponse")


@_attrs_define
class DetectResponse:
    """
    Attributes:
        x (float):
        y (float):
        z (float):
        qx (float):
        qy (float):
        qz (float):
        qw (float):
    """

    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        x = self.x

        y = self.y

        z = self.z

        qx = self.qx

        qy = self.qy

        qz = self.qz

        qw = self.qw

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "x": x,
                "y": y,
                "z": z,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "qw": qw,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        x = d.pop("x")

        y = d.pop("y")

        z = d.pop("z")

        qx = d.pop("qx")

        qy = d.pop("qy")

        qz = d.pop("qz")

        qw = d.pop("qw")

        detect_response = cls(
            x=x,
            y=y,
            z=z,
            qx=qx,
            qy=qy,
            qz=qz,
            qw=qw,
        )

        detect_response.additional_properties = d
        return detect_response

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
