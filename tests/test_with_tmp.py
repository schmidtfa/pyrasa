def test_with_temporary_files(tmp_path):
    p = tmp_path / 'hello.txt'
    p.write_text('content')
    assert p.read_text() == 'content'
    assert len(list(tmp_path.iterdir())) == 1
