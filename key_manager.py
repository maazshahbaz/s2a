#!/usr/bin/env python3
"""
API Key Management CLI Tool
Similar to OpenAI's key management interface
"""

import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import click
from tabulate import tabulate

try:
    from auth import api_key_store, APIKeyType
except ImportError as e:
    click.echo(f"❌ Import error: {e}")
    click.echo("Make sure you're running this from the s2a directory")
    sys.exit(1)


@click.group()
def cli():
    """BytePulse AI API Key Manager"""
    pass


@cli.command()
@click.option('--name', required=True, help='Name for the API key')
@click.option('--type', 'key_type', 
              type=click.Choice(['project', 'user', 'service']), 
              default='project',
              help='Type of API key')
@click.option('--rpm', default=60, help='Requests per minute limit')
@click.option('--rph', default=1000, help='Requests per hour limit')
@click.option('--rpd', default=10000, help='Requests per day limit')
@click.option('--permissions', default='transcribe,status,stats', 
              help='Comma-separated permissions')
def create(name, key_type, rpm, rph, rpd, permissions):
    """Create a new API key"""
    
    # Convert key type
    key_type_enum = {
        'project': APIKeyType.PROJECT,
        'user': APIKeyType.USER,
        'service': APIKeyType.SERVICE
    }[key_type]
    
    # Parse permissions
    perms = [p.strip() for p in permissions.split(',')]
    
    # Create key
    api_key, key_info = api_key_store.create_key(
        name=name,
        key_type=key_type_enum,
        requests_per_minute=rpm,
        requests_per_hour=rph,
        requests_per_day=rpd,
        permissions=perms
    )
    
    click.echo(f"\n✅ API Key created successfully!")
    click.echo(f"   Name: {name}")
    click.echo(f"   Key ID: {key_info.key_id}")
    click.echo(f"   Type: {key_type}")
    click.echo(f"\n🔑 API Key: {api_key}")
    click.echo(f"\n⚠️  Save this key securely - it won't be shown again!")
    click.echo(f"\n📝 Usage example:")
    click.echo(f'   curl -H "Authorization: Bearer {api_key}" \\')
    click.echo(f'        -F "audio_file=@audio.wav" \\')
    click.echo(f'        http://localhost:8001/v1/transcribe')


@cli.command()
def list():
    """List all API keys"""
    keys = api_key_store.list_keys()
    
    if not keys:
        click.echo("No API keys found.")
        return
    
    # Prepare table data
    table_data = []
    for key in keys:
        status = "🟢 Active" if key.is_active else "🔴 Revoked"
        last_used = key.last_used.strftime("%Y-%m-%d %H:%M") if key.last_used else "Never"
        
        table_data.append([
            key.key_id,
            key.name,
            key.key_type.value,
            status,
            key.usage_count,
            f"{key.total_audio_minutes:.1f}m",
            last_used,
            key.created_at.strftime("%Y-%m-%d")
        ])
    
    headers = ["Key ID", "Name", "Type", "Status", "Requests", "Audio", "Last Used", "Created"]
    
    click.echo("\n📋 API Keys:")
    click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))


@cli.command()
@click.argument('key_id')
def show(key_id):
    """Show detailed information about an API key"""
    keys = api_key_store.list_keys()
    key_info = None
    
    for key in keys:
        if key.key_id.startswith(key_id):
            key_info = key
            break
    
    if not key_info:
        click.echo(f"❌ API key not found: {key_id}")
        return
    
    click.echo(f"\n🔑 API Key Details:")
    click.echo(f"   Key ID: {key_info.key_id}")
    click.echo(f"   Name: {key_info.name}")
    click.echo(f"   Type: {key_info.key_type.value}")
    click.echo(f"   Status: {'🟢 Active' if key_info.is_active else '🔴 Revoked'}")
    click.echo(f"   Created: {key_info.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    if key_info.last_used:
        click.echo(f"   Last Used: {key_info.last_used.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    else:
        click.echo(f"   Last Used: Never")
    
    click.echo(f"\n📊 Usage Statistics:")
    click.echo(f"   Total Requests: {key_info.usage_count:,}")
    click.echo(f"   Audio Processed: {key_info.total_audio_minutes:.1f} minutes")
    
    click.echo(f"\n⚡ Rate Limits:")
    click.echo(f"   Per Minute: {key_info.requests_per_minute}")
    click.echo(f"   Per Hour: {key_info.requests_per_hour}")
    click.echo(f"   Per Day: {key_info.requests_per_day}")
    
    click.echo(f"\n🔐 Permissions:")
    for perm in key_info.permissions:
        click.echo(f"   • {perm}")


@cli.command()
@click.argument('key_id')
@click.confirmation_option(prompt='Are you sure you want to revoke this API key?')
def revoke(key_id):
    """Revoke an API key"""
    keys = api_key_store.list_keys()
    key_info = None
    
    for key in keys:
        if key.key_id.startswith(key_id):
            key_info = key
            break
    
    if not key_info:
        click.echo(f"❌ API key not found: {key_id}")
        return
    
    # We need the actual API key to revoke it
    # This is a limitation of the current storage design
    click.echo(f"❌ Key revocation requires the full API key.")
    click.echo(f"   Use the web interface or contact administrator.")


@cli.command()
def stats():
    """Show overall API usage statistics"""
    keys = api_key_store.list_keys()
    
    if not keys:
        click.echo("No API keys found.")
        return
    
    # Calculate totals
    total_keys = len(keys)
    active_keys = len([k for k in keys if k.is_active])
    total_requests = sum(k.usage_count for k in keys)
    total_audio = sum(k.total_audio_minutes for k in keys)
    
    # Most used key
    most_used = max(keys, key=lambda k: k.usage_count) if keys else None
    
    click.echo(f"\n📊 Overall Statistics:")
    click.echo(f"   Total API Keys: {total_keys}")
    click.echo(f"   Active Keys: {active_keys}")
    click.echo(f"   Total Requests: {total_requests:,}")
    click.echo(f"   Total Audio Processed: {total_audio:.1f} minutes")
    
    if most_used:
        click.echo(f"\n🏆 Most Used Key:")
        click.echo(f"   {most_used.name} ({most_used.key_id}): {most_used.usage_count:,} requests")
    
    # Key types breakdown
    type_counts = {}
    for key in keys:
        type_name = key.key_type.value
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
    
    click.echo(f"\n📈 Key Types:")
    for key_type, count in type_counts.items():
        click.echo(f"   {key_type}: {count}")


@cli.command()
def test():
    """Show test commands for API authentication"""
    click.echo("🧪 API Key Testing Guide")
    
    # Get available keys
    keys = api_key_store.list_keys()
    active_keys = [k for k in keys if k.is_active]
    
    if not active_keys:
        click.echo("❌ No active API keys found. Create one first with 'create' command.")
        return
    
    click.echo(f"\n✅ Found {len(active_keys)} active key(s)")
    click.echo("\n📝 Test Commands:")
    click.echo("\n1. Health Check:")
    click.echo('   curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:8001/health')
    
    click.echo("\n2. Transcribe Audio:")
    click.echo('   curl -H "Authorization: Bearer YOUR_API_KEY" \\')
    click.echo('        -F "audio_file=@tests/test_audio/in-9524528884-2058527609-20250125-132037-1737832837.3553.wav" \\')
    click.echo('        http://localhost:8001/v1/transcribe')
    
    click.echo("\n3. Get Stats:")
    click.echo('   curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:8001/v1/stats')
    
    click.echo("\n⚠️  Replace YOUR_API_KEY with the actual key from 'create' command")


if __name__ == '__main__':
    cli()