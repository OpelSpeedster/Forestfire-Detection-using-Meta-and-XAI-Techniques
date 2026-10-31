from src.vlm_client import parse_response

def test_parser_strict_json():
    assert parse_response('{"label":"smoke","confidence":0.8,"rationale":"thin plume"}')["label"] == "smoke"

def test_parser_with_surrounding_text():
    txt = 'Sure ```json\n{"label":"fire","confidence":0.9,"rationale":"flames visible"}\n```'
    assert parse_response(txt)["label"] == "fire"
