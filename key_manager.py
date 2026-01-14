#!/usr/bin/env python3
"""
API Key Management CLI Tool
Similar to OpenAI's key management interface - PostgreSQL backend
"""

import sys
import os
import asyncio

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import click
from tabulate import tabulate

try:
    from generated.prisma import Prisma
    from db_services.auth import PrismaAPIKeyStore, APIKeyType
except ImportError as e:
    click.echo(f"❌ Import error: {e}")
    click.echo("Make sure you're running this from the s2a directory")
    sys.exit(1)


async def get_store():
    """Get API key store with shared database connection"""
    db = Prisma()
    await db.connect()
    store = PrismaAPIKeyStore(db)
    return db, store


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
    
    async def _create():
        # Convert key type
        key_type_enum = {
            'project': APIKeyType.PROJECT,
            'user': APIKeyType.USER,
            'service': APIKeyType.SERVICE
        }[key_type]
        
        # Parse permissions
        perms = [p.strip() for p in permissions.split(',')]
        
        # Get database connection
        db, store = await get_store()
        
        try:
            # Create key
            api_key, key_info = await store.create_key(
                name=name,
                key_type=key_type_enum,
                requests_per_minute=rpm,
                requests_per_hour=rph,
                requests_per_day=rpd,
                permissions=perms
            )
            
            click.echo(f"\n✅ API Key created successfully!")
            click.echo(f"   Name: {name}")
            click.echo(f"   Key ID: {key_info.key}")
            click.echo(f"   Type: {key_type}")
            click.echo(f"\n🔑 API Key: {api_key}")
            click.echo(f"\n⚠️  Save this key securely - it won't be shown again!")
            click.echo(f"\n📝 Usage example:")
            click.echo(f'   curl -H "Authorization: Bearer {api_key}" \\')
            click.echo(f'        -F "audio_file=@audio.wav" \\')
            click.echo(f'        http://localhost:8002/v1/transcribe')
            
        finally:
            await db.disconnect()
    
    asyncio.run(_create())


@cli.command()
def list():
    """List all API keys"""
    
    async def _list():
        db, store = await get_store()
        
        try:
            keys = await store.list_keys()
            
            if not keys:
                click.echo("No API keys found.")
                return
            
            # Prepare table data
            table_data = []
            for key in keys:
                status = "🟢 Active" if key.is_active else "🔴 Revoked"
                last_used = key.last_used.strftime("%Y-%m-%d %H:%M") if key.last_used else "Never"
                
                table_data.append([
                    key.key,
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
            
        finally:
            await db.disconnect()
    
    asyncio.run(_list())


@cli.command()
@click.argument('key_id')
def show(key_id):
    """Show detailed information about an API key"""
    
    async def _show():
        db, store = await get_store()
        
        try:
            keys = await store.list_keys()
            key_info = None
            
            for key_object in keys:
                if key_object.key.startswith(key_id):
                    key_info = key_object
                    break
            
            if not key_info:
                click.echo(f"❌ API key not found: {key_id}")
                return
            
            click.echo(f"\n🔑 API Key Details:")
            click.echo(f"   Key ID: {key_info.key}")
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
                
        finally:
            await db.disconnect()
    
    asyncio.run(_show())


@cli.command()
@click.argument('key_id')
@click.confirmation_option(prompt='Are you sure you want to revoke this API key?')
def revoke(key_id):
    """Revoke an API key"""
    
    async def _revoke():
        db, store = await get_store()
        
        try:
            keys = await store.list_keys()
            key_info = None
            
            for key_object in keys:
                if key_object.key.startswith(key_id):
                    key_info = key_object
                    break
            
            if not key_info:
                click.echo(f"❌ API key not found: {key_id}")
                return
            
            # Update the key to inactive status directly in database
            await db.authkey.update(
                where={'hash': key_info.hash},
                data={'isActive': False}
            )
            
            click.echo(f"✅ API key {key_info.key} ({key_info.name}) has been revoked")
            
        finally:
            await db.disconnect()
    
    asyncio.run(_revoke())


@cli.command()
def stats():
    """Show overall API usage statistics"""
    
    async def _stats():
        db, store = await get_store()
        
        try:
            keys = await store.list_keys()
            
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
                click.echo(f"   {most_used.name} ({most_used.key}): {most_used.usage_count:,} requests")
            
            # Key types breakdown
            type_counts = {}
            for key in keys:
                type_name = key.key_type.value
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
            click.echo(f"\n📈 Key Types:")
            for key_type, count in type_counts.items():
                click.echo(f"   {key_type}: {count}")
                
        finally:
            await db.disconnect()
    
    asyncio.run(_stats())


@cli.command()
def test():
    """Show test commands for API authentication"""
    
    async def _test():
        click.echo("🧪 API Key Testing Guide")
        
        db, store = await get_store()
        
        try:
            # Get available keys
            keys = await store.list_keys()
            active_keys = [k for k in keys if k.is_active]
            
            if not active_keys:
                click.echo("❌ No active API keys found. Create one first with 'create' command.")
                return
            
            click.echo(f"\n✅ Found {len(active_keys)} active key(s)")
            click.echo("\n📝 Test Commands:")
            click.echo("\n1. Health Check:")
            click.echo('   curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:8002/v1/statistics/health')
            
            click.echo("\n2. Transcribe Audio:")
            click.echo('   curl -H "Authorization: Bearer YOUR_API_KEY" \\')
            click.echo('        -F "audio_file=@tests/test_audio/in-9524528884-2058527609-20250125-132037-1737832837.3553.wav" \\')
            click.echo('        http://localhost:8002/v1/transcribe')
            
            click.echo("\n3. Get Stats:")
            click.echo('   curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:8002/v1/stats')
            
            click.echo("\n⚠️  Replace YOUR_API_KEY with the actual key from 'create' command")
            
        finally:
            await db.disconnect()
    
    asyncio.run(_test())


if __name__ == '__main__':
    cli()