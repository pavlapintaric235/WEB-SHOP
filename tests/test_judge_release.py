"""Security boundaries for the judge sandbox; no cloud/model calls."""
import os
import unittest
from unittest.mock import patch
import auth,db,judge_release
from tests.test_workflow import WorkflowTests
from streamlit.testing.v1 import AppTest
from tools.share_judge_demo import validate_emails


class JudgeReleaseTests(unittest.TestCase):
    setUp=WorkflowTests.setUp

    def judge_users(self):
        with db.connect() as c:
            c.execute('UPDATE users SET username=? WHERE username=?',('judge-owner','owner'))
            c.execute('UPDATE users SET username=? WHERE username=?',('judge-employee','employee'))
            c.execute('UPDATE users SET password_hash=?',(auth.hash_password('a-long-test-only-password'),))

    def test_rejects_default_credentials_and_wrong_database(self):
        with patch('judge_release.sandbox_directory',return_value=self.root):
            with self.assertRaises(RuntimeError):judge_release.validate_sandbox()
            self.judge_users();judge_release.validate_sandbox()
        with self.assertRaises(RuntimeError):judge_release.validate_sandbox()

    def test_blocks_private_reference_paths(self):
        self.judge_users()
        with db.connect() as c:c.execute('UPDATE products SET image_path=? WHERE id=?',(str(self.root.parent/'private.jpg'),self.owner.products()[0]['id']))
        with patch('judge_release.sandbox_directory',return_value=self.root):
            with self.assertRaises(RuntimeError):judge_release.validate_sandbox()

    def test_gate_requires_code_and_expires(self):
        script="from judge_release import require_judge_access\nrequire_judge_access()\nimport streamlit as st\nst.success('Protected content')"
        with patch.dict(os.environ,{'WEBSHOP_JUDGE_MODE':'1','WEBSHOP_JUDGE_ACCESS_CODE':'a-separate-test-only-access-code'}):
            app=AppTest.from_string(script).run();self.assertFalse(app.success)
            app.text_input[0].input('wrong');app.button[0].click().run()
            self.assertFalse(app.success);self.assertTrue(app.error)
            app.text_input[0].input('a-separate-test-only-access-code');app.button[0].click().run()
            self.assertTrue(app.success)
            app.session_state['_judge_access_until']=1;app.run();self.assertFalse(app.success)

    def test_error_and_download_redaction_preserves_local_behavior(self):
        private=dict(source_bucket='private-bucket',items=[dict(sku='A',image_path='C:/private/photo.jpg')],url='http://private-worker')
        with patch.dict(os.environ,{'WEBSHOP_JUDGE_MODE':'1'}):
            self.assertEqual(judge_release.shareable_evidence(private),dict(items=[dict(sku='A')]))
            self.assertNotIn('private-worker',judge_release.user_error(RuntimeError('http://private-worker')))
            self.assertEqual(judge_release.user_error(ValueError('Choose a valid quantity.')),'Choose a valid quantity.')
        with patch.dict(os.environ,{'WEBSHOP_JUDGE_MODE':'0'}):self.assertIs(judge_release.shareable_evidence(private),private)

    def test_tunnel_requires_explicit_emails(self):
        self.assertEqual(validate_emails(['Judge@example.org','judge@example.org']),['judge@example.org'])
        for value in ([],['*@example.org'],[''],['not-an-email']):
            with self.assertRaises(ValueError):validate_emails(value)


if __name__=='__main__':unittest.main()
