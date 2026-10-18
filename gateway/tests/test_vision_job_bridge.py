import httpx
import pytest
from fastapi.testclient import TestClient
from laguna_gateway.tailscale_bridge import create_bridge_app


@pytest.mark.parametrize('method,path,expected',[
    ('POST','/v1/vision/jobs',202),
    ('GET','/v1/vision/jobs/vision_'+'a'*32,202),
    ('GET','/v1/vision/jobs/vision_'+'a'*64,202),
    ('GET','/v1/vision/jobs/not-valid',404),
    ('POST','/v1/vision/jobs/vision_'+'a'*32,404),
    ('GET','/v1/vision/jobs/vision_'+'a'*32+'/other',404),
])
def test_bridge_exposes_only_exact_job_routes_with_existing_auth(method,path,expected):
    calls=[]
    async def upstream(request):
        calls.append(request)
        return httpx.Response(202,json={'job_id':'fixture'})
    app=create_bridge_app(gateway_base_url='http://internal',gateway_api_key='internal-key',
        allowed_users=('owner@example.com',),upstream_transport=httpx.MockTransport(upstream))
    with TestClient(app) as client:
        response=client.request(method,path,headers={'Tailscale-User-Login':'owner@example.com'})
        assert response.status_code==expected
        if calls:
            assert calls[-1].headers['Authorization']=='Bearer internal-key'
            assert 'Tailscale-User-Login' not in calls[-1].headers
        assert client.request(method,path).status_code==401
